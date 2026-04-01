"""
Whisper Transcriber — macOS
Audio/Video to Text using OpenAI Whisper
"""

import sys
import os

if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import subprocess
import logging
import time
import math
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# ── Logging ──────────────────────────────────────────────────────────────────

log_dir = Path.home() / ".whisper_transcriber"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"transcriber_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file, encoding='utf-8')]
)
logger = logging.getLogger(__name__)

# ── FFmpeg ────────────────────────────────────────────────────────────────────

try:
    import numpy as np
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    logger.info(f"FFmpeg: {ffmpeg_exe}")
except ImportError as e:
    logger.error(f"Import error: {e}")
    ffmpeg_exe = None

# ── Whisper ───────────────────────────────────────────────────────────────────

try:
    import whisper
    import whisper.audio

    def _patched_load_audio(file, sr=16000):
        if not ffmpeg_exe:
            raise RuntimeError("FFmpeg not found.")
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")
        cmd = [
            ffmpeg_exe, "-nostdin", "-threads", "0",
            "-i", file, "-f", "s16le", "-ac", "1",
            "-acodec", "pcm_s16le", "-ar", str(sr), "-"
        ]
        process = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {err.decode(errors='ignore') if err else 'unknown'}")
        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    whisper.audio.load_audio = _patched_load_audio
    logger.info("Whisper loaded")

except ImportError as e:
    logger.error(f"Whisper import failed: {e}")
    whisper = None


# ═════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ═════════════════════════════════════════════════════════════════════════════

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG            = "#FFFFFF"   # window background — pure white
C_SURFACE       = "#F5F5F7"   # card / control surface — Apple light gray
C_TRANSCRIPT    = "#FAFAFA"   # transcript text area
C_TEXT          = "#1D1D1F"   # primary text — Apple near-black
C_SECONDARY     = "#6E6E73"   # secondary / labels
C_TERTIARY      = "#AEAEB2"   # hints, placeholders, meta
C_BORDER        = "#E5E5EA"   # subtle border
C_DIVIDER       = "#D1D1D6"   # heavier divider
C_ACCENT        = "#007AFF"   # iOS/macOS blue
C_ACCENT_HOVER  = "#0066D6"   # slightly darker blue
C_ACCENT_FG     = "#FFFFFF"   # text on accent
C_SUCCESS       = "#34C759"   # green
C_WARNING       = "#FF9F0A"   # amber
C_DANGER        = "#FF3B30"   # red

# ── Typography — SF Pro stack with Helvetica Neue fallback ────────────────────
FACE            = "SF Pro Display"  # Tk falls back gracefully on older macOS
FACE_TEXT       = "SF Pro Text"
FACE_MONO       = "SF Mono"

def font(size, weight="normal", face=None):
    f = face or FACE_TEXT
    if weight == "bold":
        return (f, size, "bold")
    return (f, size)

FONT_DISPLAY    = (FACE, 20, "bold")       # app title
FONT_HEADLINE   = (FACE_TEXT, 13, "bold")  # section headers
FONT_BODY       = (FACE_TEXT, 13)          # primary UI text
FONT_SUBHEAD    = (FACE_TEXT, 11)          # secondary labels, captions
FONT_CAPTION    = (FACE_TEXT, 10)          # hints, meta, tiny labels
FONT_MONO       = (FACE_MONO, 12)          # transcript
FONT_MONO_SM    = (FACE_MONO, 11)          # elapsed clock

# ── Data ──────────────────────────────────────────────────────────────────────

SUPPORTED_FORMATS = (
    ".mp3", ".wav", ".m4a", ".ogg", ".flac",
    ".mp4", ".webm", ".mkv", ".avi", ".mov"
)

MODEL_INFO = {
    "tiny":   {"size": "39 MB",  "speed": "~1×",  "quality": "Basic",  "desc": "Quick drafts"},
    "base":   {"size": "74 MB",  "speed": "~3×",  "quality": "Good",   "desc": "Everyday use"},
    "small":  {"size": "244 MB", "speed": "~6×",  "quality": "Better", "desc": "Accurate"},
    "medium": {"size": "769 MB", "speed": "~18×", "quality": "Great",  "desc": "High accuracy"},
    "large":  {"size": "1.5 GB", "speed": "~32×", "quality": "Best",   "desc": "Maximum quality"},
}

LANGUAGES = {
    "Auto-detect": None,
    "English": "en", "Finnish": "fi", "Swedish": "sv",
    "German": "de",  "French": "fr",  "Spanish": "es",
    "Italian": "it", "Dutch": "nl",   "Polish": "pl",
    "Portuguese": "pt", "Russian": "ru",
    "Mandarin Chinese": "zh",
    "Japanese": "ja", "Korean": "ko",
}

OUTPUT_MODES = {
    "Transcribe (original language)": "source",
    "Translate to English": "en",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def validate_file(filepath):
    if not filepath:
        return False, "No file selected."
    p = Path(filepath)
    if not p.exists():
        return False, f"File not found: {filepath}"
    if not p.is_file():
        return False, f"Not a file: {filepath}"
    if p.suffix.lower() not in SUPPORTED_FORMATS:
        return False, f"Unsupported format: {p.suffix}"
    return True, "OK"

def format_duration(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"

def format_clock(seconds):
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

def format_file_size(size_bytes):
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def get_file_duration(filepath):
    if not ffmpeg_exe:
        return None
    try:
        result = subprocess.run(
            [ffmpeg_exe, "-i", filepath, "-hide_banner"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20
        )
        for line in (result.stderr or b"").decode("utf-8", errors="ignore").split('\n'):
            if 'Duration:' in line:
                t = line.split('Duration:')[1].split(',')[0].strip()
                h, m, s = t.split(':')
                return float(h) * 3600 + float(m) * 60 + float(s)
    except Exception as e:
        logger.warning(f"Duration check failed: {e}")
    return None


# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM WIDGETS
# ═════════════════════════════════════════════════════════════════════════════

class Button(tk.Label):
    """
    Flat, hover-aware button.
    primary=True  → filled accent blue
    primary=False → transparent with hover fill
    """
    def __init__(self, parent, text, command,
                 primary=False, danger=False,
                 font=FONT_BODY, padx=16, pady=6, **kw):

        if danger:
            self._n_bg, self._n_fg = C_DANGER,   C_ACCENT_FG
            self._h_bg, self._h_fg = "#E0362B",  C_ACCENT_FG
        elif primary:
            self._n_bg, self._n_fg = C_ACCENT,       C_ACCENT_FG
            self._h_bg, self._h_fg = C_ACCENT_HOVER,  C_ACCENT_FG
        else:
            self._n_bg, self._n_fg = C_SURFACE,   C_TEXT
            self._h_bg, self._h_fg = C_BORDER,    C_TEXT

        self._cmd = command
        self._off = False

        super().__init__(parent, text=text, font=font,
                         bg=self._n_bg, fg=self._n_fg,
                         padx=padx, pady=pady,
                         cursor="arrow", **kw)

        self.bind("<Enter>",    self._enter)
        self.bind("<Leave>",    self._leave)
        self.bind("<Button-1>", self._click)

    def _enter(self, _=None):
        if not self._off:
            self.config(bg=self._h_bg, fg=self._h_fg)

    def _leave(self, _=None):
        if not self._off:
            self.config(bg=self._n_bg, fg=self._n_fg)

    def _click(self, _=None):
        if not self._off and self._cmd:
            self._cmd()

    def disable(self):
        self._off = True
        self.config(bg=C_SURFACE, fg=C_TERTIARY, cursor="")

    def enable(self):
        self._off = False
        self.config(bg=self._n_bg, fg=self._n_fg, cursor="arrow")


class SegmentedControl(tk.Frame):
    """
    Two-option segmented pill control (like iOS/macOS segmented pickers).
    Calls callback(value) on change.
    """
    def __init__(self, parent, options, variable, callback=None, **kw):
        super().__init__(parent, bg=C_SURFACE,
                         highlightthickness=1, highlightbackground=C_BORDER,
                         **kw)
        self._var = variable
        self._callback = callback
        self._btns = {}

        for i, (label, value) in enumerate(options):
            btn = tk.Label(
                self, text=label,
                font=FONT_CAPTION,
                padx=12, pady=4,
                cursor="arrow"
            )
            btn.grid(row=0, column=i, sticky="nsew")
            self.columnconfigure(i, weight=1)
            btn.bind("<Button-1>", lambda e, v=value: self._select(v))
            self._btns[value] = btn

        self._refresh()
        variable.trace_add("write", lambda *_: self._refresh())

    def _select(self, value):
        self._var.set(value)
        if self._callback:
            self._callback(value)

    def _refresh(self):
        current = self._var.get()
        for value, btn in self._btns.items():
            if value == current:
                btn.config(bg=C_ACCENT, fg=C_ACCENT_FG)
            else:
                btn.config(bg=C_SURFACE, fg=C_SECONDARY)


class ToolTip:
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip = None
        self._id = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._cancel)
        widget.bind("<ButtonPress>", self._cancel)

    def _schedule(self, e=None):
        self._cancel()
        self._id = self.widget.after(self.delay, self._show)

    def _show(self):
        x = self.widget.winfo_rootx() + 8
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        frame = tk.Frame(tw, bg="#1C1C1E", padx=1, pady=1)
        frame.pack()
        tk.Label(frame, text=self.text, justify="left",
                 bg="#2C2C2E", fg="#F2F2F7",
                 font=(FACE_TEXT, 11),
                 padx=10, pady=7, wraplength=300).pack()

    def _cancel(self, e=None):
        if self._id:
            self.widget.after_cancel(self._id)
            self._id = None
        if self.tip:
            self.tip.destroy()
            self.tip = None


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

class WhisperTranscriber:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.model = None
        self.current_model_name = None
        self.is_loading_model = False
        self.is_transcribing = False
        self._is_closing = False
        self._transcription_start = None
        self._elapsed_timer_id = None
        self._file_duration = None

        self._setup_window()
        self._build_menubar()

        if not self._check_deps():
            self.root.after(0, self.root.destroy)
            return

        self._build_ui()
        logger.info("App ready")

    # ── Window ────────────────────────────────────────────────────────────────

    def _setup_window(self):
        self.root.title("Whisper Transcriber")
        self.root.geometry("760x720")
        self.root.minsize(640, 580)
        self.root.resizable(True, True)
        self.root.configure(bg=C_BG)

        self.root.update_idletasks()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"+{(sw - 760) // 2}+{(sh - 720) // 2}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ttk style — minimal, just for progress bar and scrollbar
        style = ttk.Style()
        try:
            style.theme_use("aqua")
        except tk.TclError:
            pass
        style.configure("Accent.Horizontal.TProgressbar",
                         thickness=3,
                         troughcolor=C_BORDER,
                         background=C_ACCENT)
        style.configure("Transcript.Vertical.TScrollbar",
                         troughcolor=C_TRANSCRIPT,
                         background=C_BORDER)

    def _build_menubar(self):
        menubar = tk.Menu(self.root)

        app_menu = tk.Menu(menubar, name="apple", tearoff=False)
        menubar.add_cascade(menu=app_menu)
        app_menu.add_command(label="About Whisper Transcriber", command=self._show_about)

        file_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open…",            accelerator="⌘O", command=self._browse_file)
        file_menu.add_separator()
        file_menu.add_command(label="Save Transcript…", accelerator="⌘S", command=self._save_transcript)
        file_menu.add_separator()
        file_menu.add_command(label="Close Window",     accelerator="⌘W", command=self._on_close)

        edit_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Copy Transcript",  accelerator="⌘⇧C", command=self._copy_transcript)
        edit_menu.add_command(label="Clear Transcript",               command=self._clear_output)

        help_menu = tk.Menu(menubar, name="help", tearoff=False)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Whisper Transcriber Help", command=self._show_help)

        self.root.configure(menu=menubar)
        self.root.bind_all("<Command-o>",       lambda e: self._browse_file())
        self.root.bind_all("<Command-s>",       lambda e: self._save_transcript())
        self.root.bind_all("<Command-Shift-c>", lambda e: self._copy_transcript())
        self.root.bind_all("<Command-Shift-C>", lambda e: self._copy_transcript())
        self.root.bind_all("<Command-w>",       lambda e: self._on_close())
        self.root.bind_all("<Command-Return>",  lambda e: self._transcribe())

        try:
            self.root.createcommand("tk::mac::Quit", self._on_close)
        except Exception:
            pass

    def _check_deps(self):
        errors = []
        if not ffmpeg_exe:
            errors.append("imageio-ffmpeg not found")
        if whisper is None:
            errors.append("openai-whisper not found")
        if errors:
            messagebox.showerror("Cannot Start",
                "Missing dependencies:\n\n" +
                "\n".join(f"  • {e}" for e in errors) +
                "\n\nInstall with:\n  pip install openai-whisper imageio-ffmpeg")
            return False
        return True

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Outer padded container
        outer = tk.Frame(self.root, bg=C_BG)
        outer.pack(fill="both", expand=True, padx=28, pady=(20, 0))

        self._build_header(outer)
        self._build_settings_card(outer)
        self._build_file_zone(outer)
        self._build_action_row(outer)
        self._build_transcript(outer)
        self._build_statusbar()

    def _card(self, parent, pady=(12, 0)):
        """A surface-colored rounded card with a subtle border."""
        frame = tk.Frame(parent,
                         bg=C_SURFACE,
                         highlightthickness=1,
                         highlightbackground=C_BORDER)
        frame.pack(fill="x", pady=pady)
        inner = tk.Frame(frame, bg=C_SURFACE)
        inner.pack(fill="x", padx=16, pady=12)
        return inner

    def _divider(self, parent, pady=0):
        tk.Frame(parent, bg=C_BORDER, height=1).pack(fill="x", pady=pady)

    # ── Header ────────────────────────────────────────────────────────────────

    def _build_header(self, parent):
        row = tk.Frame(parent, bg=C_BG)
        row.pack(fill="x", pady=(0, 16))

        tk.Label(row, text="Whisper Transcriber",
                 font=FONT_DISPLAY, fg=C_TEXT, bg=C_BG, anchor="w").pack(side="left")

        tk.Label(row, text="Local  ·  Private  ·  No cloud",
                 font=FONT_CAPTION, fg=C_TERTIARY, bg=C_BG).pack(side="right", anchor="s", pady=(0, 4))

    # ── Settings card ─────────────────────────────────────────────────────────

    def _build_settings_card(self, parent):
        card = self._card(parent, pady=(0, 0))

        # Row 1: Model
        self._build_model_row(card)

        # Intra-card divider
        self._divider(card, pady=(10, 10))

        # Row 2: Language  |  Output mode  |  Timestamps
        opts_row = tk.Frame(card, bg=C_SURFACE)
        opts_row.pack(fill="x")

        # Language
        self._add_combo_group(
            opts_row,
            label="Language",
            var_attr="lang_var",
            default="Auto-detect",
            values=list(LANGUAGES.keys()),
            width=14,
            tooltip="Auto-detect works well for most audio.\nSet manually for better accuracy."
        )
        self._vsep(opts_row)

        # Output mode
        self._add_combo_group(
            opts_row,
            label="Output",
            var_attr="output_mode_var",
            default="Transcribe (original language)",
            values=list(OUTPUT_MODES.keys()),
            width=26,
            tooltip="Transcribe keeps the original language.\nTranslate outputs English text via Whisper."
        )
        self._vsep(opts_row)

        # Timestamps — segmented control
        ts_grp = tk.Frame(opts_row, bg=C_SURFACE)
        ts_grp.pack(side="left")

        tk.Label(ts_grp, text="Format",
                 font=FONT_CAPTION, fg=C_SECONDARY, bg=C_SURFACE).pack(anchor="w")

        self.timestamps_var = tk.StringVar(value="plain")
        seg = SegmentedControl(
            ts_grp,
            options=[("Plain", "plain"), ("Timestamps", "ts")],
            variable=self.timestamps_var
        )
        seg.pack(anchor="w", pady=(4, 0))

    def _build_model_row(self, parent):
        row = tk.Frame(parent, bg=C_SURFACE)
        row.pack(fill="x")

        # Label
        tk.Label(row, text="Model",
                 font=FONT_CAPTION, fg=C_SECONDARY, bg=C_SURFACE, width=6, anchor="w"
                 ).pack(side="left")

        # Combobox
        self.model_var = tk.StringVar(value="base")
        self.model_combo = ttk.Combobox(
            row, textvariable=self.model_var,
            values=list(MODEL_INFO.keys()), width=9, state="readonly"
        )
        self.model_combo.pack(side="left", padx=(0, 8))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        # Load button
        self.load_btn = Button(row, text="Load", command=self._load_model,
                               primary=True, padx=14, pady=4, font=FONT_CAPTION)
        self.load_btn.pack(side="left")

        # Status dot + label
        self._dot = tk.Label(row, text="●", font=(FACE_TEXT, 10),
                             fg=C_BORDER, bg=C_SURFACE)
        self._dot.pack(side="left", padx=(12, 0))

        self._model_status_lbl = tk.Label(row, text="Not loaded",
                                          font=FONT_CAPTION, fg=C_SECONDARY, bg=C_SURFACE)
        self._model_status_lbl.pack(side="left", padx=(3, 0))

        # Descriptor (right-aligned)
        info = MODEL_INFO["base"]
        self._model_desc = tk.Label(
            row,
            text=f"{info['size']}  ·  {info['speed']} realtime  ·  {info['quality']}",
            font=FONT_CAPTION, fg=C_TERTIARY, bg=C_SURFACE
        )
        self._model_desc.pack(side="right")

    def _add_combo_group(self, parent, label, var_attr, default, values, width, tooltip=None):
        grp = tk.Frame(parent, bg=C_SURFACE)
        grp.pack(side="left", padx=(0, 0))

        tk.Label(grp, text=label, font=FONT_CAPTION,
                 fg=C_SECONDARY, bg=C_SURFACE).pack(anchor="w")

        var = tk.StringVar(value=default)
        setattr(self, var_attr, var)
        cb = ttk.Combobox(grp, textvariable=var, values=values,
                          width=width, state="readonly")
        cb.pack(anchor="w", pady=(4, 0))
        if tooltip:
            ToolTip(cb, tooltip)

    def _vsep(self, parent, pad=20):
        tk.Frame(parent, bg=C_SURFACE, width=pad).pack(side="left")

    def _on_model_change(self, e=None):
        name = self.model_var.get()
        info = MODEL_INFO.get(name, {})
        self._model_desc.config(
            text=f"{info['size']}  ·  {info['speed']} realtime  ·  {info['quality']}"
        )

    # ── File zone ─────────────────────────────────────────────────────────────

    def _build_file_zone(self, parent):
        self._fz_outer = tk.Frame(parent, bg=C_BG)
        self._fz_outer.pack(fill="x", pady=(12, 0))

        # Drop canvas
        self._drop_canvas = tk.Canvas(
            self._fz_outer, height=86,
            bg=C_SURFACE,
            highlightthickness=1,
            highlightbackground=C_BORDER,
            cursor="pointinghand"
        )
        self._drop_canvas.pack(fill="x")
        self._drop_canvas.bind("<Configure>", self._draw_drop)
        self._drop_canvas.bind("<Button-1>",  lambda e: self._browse_file())
        self._drop_canvas.bind("<Enter>",     self._drop_hover_on)
        self._drop_canvas.bind("<Leave>",     self._drop_hover_off)

        # File info strip (hidden until a file is selected)
        self._file_strip = tk.Frame(self._fz_outer, bg=C_BG)

        si = tk.Frame(self._file_strip, bg=C_BG)
        si.pack(fill="x", pady=(10, 0))

        self._file_icon = tk.Label(si, text="♫", font=(FACE_TEXT, 18),
                                   fg=C_ACCENT, bg=C_BG)
        self._file_icon.pack(side="left")

        tcol = tk.Frame(si, bg=C_BG)
        tcol.pack(side="left", padx=(10, 0), fill="x", expand=True)

        self._file_name_lbl = tk.Label(tcol, text="", font=FONT_BODY,
                                       fg=C_TEXT, bg=C_BG, anchor="w")
        self._file_name_lbl.pack(fill="x")

        self._file_meta_lbl = tk.Label(tcol, text="", font=FONT_CAPTION,
                                       fg=C_SECONDARY, bg=C_BG, anchor="w")
        self._file_meta_lbl.pack(fill="x")

        Button(si, text="Change…", command=self._browse_file,
               padx=12, pady=4, font=FONT_CAPTION).pack(side="right")

        self.file_path = tk.StringVar()

        try:
            self.root.drop_target_register("DND_Files")
            self.root.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            pass

    def _draw_drop(self, event=None):
        c = self._drop_canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        cx, cy = w // 2, h // 2

        pad = 8
        c.create_rectangle(pad, pad, w - pad, h - pad,
                           outline=C_DIVIDER, dash=(6, 4), width=1)

        c.create_text(cx, cy - 10, text="↓",
                      font=(FACE, 20), fill=C_TERTIARY, anchor="center")
        c.create_text(cx, cy + 16, text="Drop a file here, or click to browse",
                      font=(FACE_TEXT, 12), fill=C_SECONDARY, anchor="center")
        c.create_text(cx, cy + 34, text="mp3  ·  wav  ·  m4a  ·  mp4  ·  mov  ·  and more",
                      font=(FACE_TEXT, 10), fill=C_TERTIARY, anchor="center")

    def _drop_hover_on(self, e=None):
        self._drop_canvas.configure(bg="#EDF4FF", highlightbackground=C_ACCENT)

    def _drop_hover_off(self, e=None):
        self._drop_canvas.configure(bg=C_SURFACE, highlightbackground=C_BORDER)

    def _on_drop(self, event):
        if self.is_transcribing or self.is_loading_model:
            self._set_status("Busy — please wait")
            return
        try:
            paths = self.root.tk.splitlist(event.data)
        except Exception:
            paths = [event.data]
        if paths:
            self._set_file(paths[0].strip().strip("{}"))

    def _show_file_strip(self, name, meta):
        ext = Path(name).suffix.lower()
        self._file_icon.config(
            text="▶" if ext in (".mp4", ".webm", ".mkv", ".avi", ".mov") else "♫"
        )
        self._file_name_lbl.config(text=name)
        self._file_meta_lbl.config(text=meta)
        self._drop_canvas.pack_forget()
        self._file_strip.pack(fill="x")

    def _reset_file_zone(self):
        self._file_strip.pack_forget()
        self._drop_canvas.config(height=86, bg=C_SURFACE, highlightbackground=C_BORDER)
        self._drop_canvas.pack(fill="x")

    # ── Action row ────────────────────────────────────────────────────────────

    def _build_action_row(self, parent):
        row = tk.Frame(parent, bg=C_BG)
        row.pack(fill="x", pady=(14, 0))

        self.transcribe_btn = Button(
            row, text="Transcribe", command=self._transcribe,
            primary=True, padx=22, pady=8, font=(FACE_TEXT, 13, "bold")
        )
        self.transcribe_btn.pack(side="left")
        ToolTip(self.transcribe_btn, "Start transcription  ⌘↩")

        self.cancel_btn = Button(row, text="Cancel", command=self._cancel,
                                 padx=14, pady=8, font=FONT_BODY)

        self.progress = ttk.Progressbar(
            row, mode="determinate", length=100,
            style="Accent.Horizontal.TProgressbar"
        )

        self._elapsed_lbl = tk.Label(row, text="", font=FONT_MONO_SM,
                                     fg=C_SECONDARY, bg=C_BG)

        self._action_info = tk.Label(row, text="", font=FONT_CAPTION,
                                     fg=C_SECONDARY, bg=C_BG)

    # ── Transcript ────────────────────────────────────────────────────────────

    def _build_transcript(self, parent):
        # Toolbar row
        bar = tk.Frame(parent, bg=C_BG)
        bar.pack(fill="x", pady=(20, 6))

        tk.Label(bar, text="Transcript", font=FONT_HEADLINE,
                 fg=C_TEXT, bg=C_BG).pack(side="left")

        self._word_count = tk.Label(bar, text="", font=FONT_CAPTION,
                                    fg=C_TERTIARY, bg=C_BG)
        self._word_count.pack(side="right")

        Button(bar, text="Clear", command=self._clear_output,
               padx=10, pady=3, font=FONT_CAPTION
               ).pack(side="right", padx=(4, 8))
        Button(bar, text="Copy", command=self._copy_transcript,
               padx=10, pady=3, font=FONT_CAPTION
               ).pack(side="right", padx=(4, 0))
        Button(bar, text="Save…", command=self._save_transcript,
               padx=10, pady=3, font=FONT_CAPTION
               ).pack(side="right", padx=(4, 0))

        # Text area — bordered card
        border = tk.Frame(parent, bg=C_BORDER, highlightthickness=0)
        border.pack(fill="both", expand=True)
        inner = tk.Frame(border, bg=C_TRANSCRIPT)
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        self.output = tk.Text(
            inner,
            font=FONT_MONO,
            wrap=tk.WORD,
            relief="flat", borderwidth=0,
            bg=C_TRANSCRIPT,
            fg=C_TEXT,
            insertbackground=C_TEXT,
            selectbackground=C_ACCENT,
            selectforeground=C_ACCENT_FG,
            padx=16, pady=14,
            spacing1=2, spacing2=4,
            undo=True
        )
        sb = ttk.Scrollbar(inner, orient="vertical",
                           command=self.output.yview,
                           style="Transcript.Vertical.TScrollbar")
        self.output.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.output.pack(side="left", fill="both", expand=True)
        self.output.bind("<<Modified>>", self._on_text_modified)

        self._show_placeholder()

    def _show_placeholder(self):
        self.output.config(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.insert("1.0", "Transcript will appear here…")
        self.output.config(fg=C_TERTIARY)
        self._placeholder_active = True

    def _clear_placeholder(self):
        if getattr(self, '_placeholder_active', False):
            self.output.config(state="normal")
            self.output.delete("1.0", tk.END)
            self.output.config(fg=C_TEXT)
            self._placeholder_active = False

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        tk.Frame(self.root, bg=C_BORDER, height=1).pack(fill="x", side="bottom")
        bar = tk.Frame(self.root, bg=C_SURFACE)
        bar.pack(fill="x", side="bottom")

        self._status_var = tk.StringVar(value="Ready")
        tk.Label(bar, textvariable=self._status_var,
                 font=FONT_CAPTION, fg=C_SECONDARY, bg=C_SURFACE, anchor="w"
                 ).pack(side="left", padx=14, pady=(5, 6))

        tk.Label(bar, text="Local only  ·  No cloud",
                 font=FONT_CAPTION, fg=C_TERTIARY, bg=C_SURFACE
                 ).pack(side="right", padx=14, pady=(5, 6))

    # ── Text change handler ───────────────────────────────────────────────────

    def _on_text_modified(self, e=None):
        if self.output.edit_modified():
            if not getattr(self, '_placeholder_active', False):
                text = self.output.get("1.0", tk.END).strip()
                self._word_count.config(
                    text=f"{len(text.split()):,} words" if text else ""
                )
            self.output.edit_modified(False)

    # ── Load model ────────────────────────────────────────────────────────────

    def _load_model(self):
        if self.is_loading_model:
            return
        if self.is_transcribing:
            messagebox.showinfo("Busy", "Please wait for transcription to finish.")
            return

        name = self.model_var.get()
        if self.model and self.current_model_name == name:
            self._set_status(f"{name.capitalize()} already loaded")
            return

        self.is_loading_model = True

        def task():
            try:
                self._set_busy(True, f"Loading {name}…")
                self._run_on_ui(lambda: self._set_dot(C_WARNING, f"Loading {name}…"))
                t0 = time.time()
                self.model = whisper.load_model(name)
                self.current_model_name = name
                elapsed = time.time() - t0
                self._run_on_ui(lambda: self._set_dot(C_SUCCESS, f"{name.capitalize()} ready"))
                self._set_status(f"{name.capitalize()} loaded  ({elapsed:.1f}s)")
            except Exception as e:
                logger.error(f"Load failed: {e}")
                self._run_on_ui(lambda: self._set_dot(C_DANGER, "Load failed"))
                self._run_on_ui(lambda: messagebox.showerror("Model Error",
                                    f"Could not load model:\n\n{e}"))
                self._set_status("Error loading model")
            finally:
                self.is_loading_model = False
                self._set_busy(False)

        threading.Thread(target=task, daemon=True).start()

    # ── Browse / drop ─────────────────────────────────────────────────────────

    def _browse_file(self):
        if self.is_transcribing or self.is_loading_model:
            return
        path = filedialog.askopenfilename(
            title="Open Audio or Video File",
            filetypes=[
                ("Audio/Video", " ".join(f"*{e}" for e in SUPPORTED_FORMATS)),
                ("Audio",  "*.mp3 *.wav *.m4a *.ogg *.flac"),
                ("Video",  "*.mp4 *.webm *.mkv *.avi *.mov"),
                ("All",    "*.*"),
            ]
        )
        if path:
            self._set_file(path)

    def _set_file(self, filepath):
        ok, msg = validate_file(filepath)
        if not ok:
            messagebox.showerror("Invalid File", msg)
            return
        try:
            self.file_path.set(filepath)
            p = Path(filepath)
            size = format_file_size(p.stat().st_size)
            duration = get_file_duration(filepath)
            self._file_duration = duration
            meta = "  ·  ".join(filter(None, [
                p.suffix.upper().lstrip("."),
                size,
                format_duration(duration) if duration else None
            ]))
            self._show_file_strip(p.name, meta)
            self._set_status(f"Ready — {p.name}")
            logger.info(f"File: {filepath}")
        except OSError as e:
            messagebox.showerror("File Error", str(e))
            self.file_path.set("")
            self._file_duration = None

    # ── Transcribe ────────────────────────────────────────────────────────────

    def _transcribe(self):
        if self.is_transcribing or self.is_loading_model:
            return
        if not self.model:
            messagebox.showwarning("No Model", "Load a model first.")
            return

        filepath = self.file_path.get()
        ok, _ = validate_file(filepath)
        if not ok:
            messagebox.showwarning("No File", "Select an audio or video file first.")
            return

        language         = LANGUAGES.get(self.lang_var.get())
        with_timestamps  = self.timestamps_var.get() == "ts"
        output_mode_disp = self.output_mode_var.get()
        whisper_task     = "translate" if OUTPUT_MODES.get(output_mode_disp) == "en" else "transcribe"

        self.is_transcribing = True
        self._transcription_start = time.time()

        def task():
            try:
                self._set_busy(True, "Transcribing…")
                self._run_on_ui(self._show_transcription_ui)
                result = self._transcribe_live(filepath, language, whisper_task, with_timestamps)
                elapsed = time.time() - self._transcription_start

                if not self.is_transcribing:
                    return

                text = self._format_ts(result) if with_timestamps else result["text"].strip()
                lang = result.get("language", "?")
                self._run_on_ui(lambda: self._show_result(text, elapsed, lang, output_mode_disp))

            except Exception as e:
                logger.error(f"Transcription error: {e}")
                self._run_on_ui(lambda: messagebox.showerror("Transcription Failed", str(e)))
                self._run_on_ui(lambda: self._set_status("Transcription failed"))
            finally:
                self.is_transcribing = False
                self._run_on_ui(lambda: self._set_busy(False))
                self._run_on_ui(self._hide_transcription_ui)

        threading.Thread(target=task, daemon=True).start()

    def _transcribe_live(self, filepath, language, task_name, with_timestamps):
        self._run_on_ui(lambda: (self._clear_placeholder(),
                                 self.output.delete("1.0", tk.END)))

        audio = whisper.audio.load_audio(filepath)
        sr = whisper.audio.SAMPLE_RATE
        total_s = len(audio) / sr if len(audio) else 0
        n_chunks = max(1, math.ceil(len(audio) / (sr * 30)))

        segments, texts, detected = [], [], language

        for i in range(n_chunks):
            if not self.is_transcribing:
                break
            s0 = i * sr * 30
            s1 = min(len(audio), s0 + sr * 30)
            chunk = audio[s0:s1]
            if not len(chunk):
                continue

            res = self.model.transcribe(chunk, language=detected,
                                        task=task_name, verbose=False)
            if not detected:
                detected = res.get("language")

            offset = s0 / sr
            for seg in res.get("segments", []):
                adj = dict(seg)
                adj["start"] = seg["start"] + offset
                adj["end"]   = seg["end"]   + offset
                segments.append(adj)

            t = res.get("text", "").strip()
            if t:
                texts.append(t)

            processed = min(s1 / sr, total_s)
            preview = "\n\n".join(
                f"[{format_duration(s['start'])} → {format_duration(s['end'])}]  {s['text'].strip()}"
                for s in segments if s.get("text", "").strip()
            ) if with_timestamps else "\n\n".join(texts)

            self._run_on_ui(
                lambda txt=preview, p=processed, tot=total_s:
                    self._live_update(txt, p, tot)
            )

        return {"text": " ".join(texts).strip(),
                "segments": segments,
                "language": detected or "?"}

    def _live_update(self, text, processed, total):
        self._clear_placeholder()
        self.output.delete("1.0", tk.END)
        if text:
            self.output.insert("1.0", text)
            self.output.see(tk.END)
        self.output.edit_modified(True)

        total_ = max(total or processed or 1, 1)
        pct = max(0, min(100, int(processed / total_ * 100)))
        self.progress.configure(mode="determinate", maximum=100, value=pct)

        label = (f"Transcribing  {pct}%  ·  "
                 f"{format_duration(processed)} / {format_duration(total) if total else '?'}")
        self._action_info.config(text=label)
        self._set_status(label)

    def _format_ts(self, result):
        return "\n\n".join(
            f"[{format_duration(s['start'])} → {format_duration(s['end'])}]  {s['text'].strip()}"
            for s in result.get("segments", [])
        )

    def _show_result(self, text, elapsed, lang, mode_disp):
        self._clear_placeholder()
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)
        label = "Translated" if "Translate" in mode_disp else "Transcribed"
        self._set_status(f"{label} in {format_duration(elapsed)}  ·  Language: {lang}")

    def _cancel(self):
        if self.is_transcribing:
            self.is_transcribing = False
            self._set_status("Cancelled")

    # ── Transcription UI state ────────────────────────────────────────────────

    def _show_transcription_ui(self):
        self.cancel_btn.pack(side="left", padx=(10, 0))
        self.progress.configure(mode="determinate", maximum=100, value=0)
        self.progress.pack(side="left", padx=(14, 0), pady=0)
        self._elapsed_lbl.config(text="00:00")
        self._elapsed_lbl.pack(side="left", padx=(12, 0))
        self._action_info.config(text="Preparing…")
        self._action_info.pack(side="left", padx=(10, 0))
        self._start_elapsed()

    def _hide_transcription_ui(self):
        self._stop_elapsed()
        self.cancel_btn.pack_forget()
        self.progress.pack_forget()
        self._elapsed_lbl.pack_forget()
        self._action_info.pack_forget()

    def _start_elapsed(self):
        self._stop_elapsed()
        def tick():
            if self.is_transcribing and self._transcription_start:
                self._elapsed_lbl.config(
                    text=format_clock(time.time() - self._transcription_start)
                )
                self._elapsed_timer_id = self.root.after(1000, tick)
        tick()

    def _stop_elapsed(self):
        if self._elapsed_timer_id:
            self.root.after_cancel(self._elapsed_timer_id)
            self._elapsed_timer_id = None

    # ── Transcript actions ────────────────────────────────────────────────────

    def _save_transcript(self):
        text = self._tx()
        if not text:
            messagebox.showwarning("Nothing to Save", "Transcript is empty.")
            return
        default = "transcript.txt"
        if self.file_path.get():
            stem = Path(self.file_path.get()).stem
            suffix = "_english_translation" if OUTPUT_MODES.get(self.output_mode_var.get()) == "en" \
                     else "_transcript"
            default = stem + suffix + ".txt"
        dest = filedialog.asksaveasfilename(
            title="Save Transcript", defaultextension=".txt",
            initialfile=default,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if dest:
            try:
                Path(dest).write_text(text, encoding="utf-8")
                self._set_status(f"Saved: {Path(dest).name}")
            except Exception as e:
                messagebox.showerror("Save Failed", str(e))

    def _copy_transcript(self):
        text = self._tx()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status("Copied to clipboard")

    def _clear_output(self):
        if not self._tx():
            return
        self._show_placeholder()
        self._word_count.config(text="")
        self._set_status("Cleared")

    def _tx(self):
        if getattr(self, '_placeholder_active', False):
            return ""
        return self.output.get("1.0", tk.END).strip()

    # ── About / Help ──────────────────────────────────────────────────────────

    def _show_help(self):
        win = tk.Toplevel(self.root)
        win.title("Help")
        win.geometry("480x460")
        win.configure(bg=C_BG)
        win.resizable(False, False)

        frame = tk.Frame(win, bg=C_BG)
        frame.pack(fill="both", expand=True, padx=28, pady=24)

        sections = [
            ("Getting Started",
             "1. Select a model and click Load.\n"
             "   Base is a good starting point.\n\n"
             "2. Drop a file onto the drop zone, or click it to browse.\n\n"
             "3. Set language if auto-detect isn't accurate.\n\n"
             "4. Click Transcribe, or press ⌘↩.\n\n"
             "5. Copy, save, or edit the transcript."),
            ("Models",
             "tiny    39 MB   Fastest   Basic accuracy\n"
             "base    74 MB   Fast      Good  (recommended start)\n"
             "small  244 MB   Medium    Better\n"
             "medium 769 MB   Slow      Great\n"
             "large  1.5 GB   Slowest   Best"),
            ("Privacy",
             "Audio and video never leave your Mac.\n"
             "Whisper runs entirely on this machine.\n"
             "Internet is only needed to download a model once."),
        ]

        for title, body in sections:
            tk.Label(frame, text=title, font=FONT_HEADLINE,
                     fg=C_TEXT, bg=C_BG, anchor="w").pack(fill="x", pady=(14, 4))
            tk.Label(frame, text=body, font=FONT_BODY,
                     fg=C_TEXT, bg=C_BG, anchor="w",
                     justify="left", wraplength=420).pack(fill="x")

        Button(frame, text="Close", command=win.destroy,
               padx=14, pady=6, font=FONT_BODY).pack(pady=(20, 0))

    def _show_about(self):
        messagebox.showinfo(
            "Whisper Transcriber",
            "Whisper Transcriber\n\n"
            "Local audio and video transcription\npowered by OpenAI Whisper.\n\n"
            "Runs entirely on your Mac.\n"
            "Audio is never uploaded to OpenAI."
        )

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _set_busy(self, busy, msg=None):
        def apply():
            if busy:
                self.transcribe_btn.disable()
                self.load_btn.disable()
            else:
                self.progress.configure(mode="determinate", maximum=100, value=0)
                self.transcribe_btn.enable()
                self.load_btn.enable()
        self._run_on_ui(apply)
        if msg:
            self._set_status(msg)

    def _set_status(self, msg):
        self._run_on_ui(lambda: (
            hasattr(self, "_status_var") and self._status_var.set(msg)
        ))
        logger.info(f"Status: {msg}")

    def _set_dot(self, color, text):
        def apply():
            if hasattr(self, "_dot"):
                self._dot.config(fg=color)
                self._model_status_lbl.config(text=text)
        self._run_on_ui(apply)

    def _run_on_ui(self, callback):
        try:
            if self._is_closing:
                return
            if threading.current_thread() is threading.main_thread():
                callback()
            elif self.root.winfo_exists():
                self.root.after(0, callback)
        except tk.TclError:
            pass

    def _on_close(self):
        self._is_closing = True
        self.is_transcribing = False
        self.is_loading_model = False
        self._stop_elapsed()
        try:
            self.root.destroy()
        except tk.TclError:
            pass


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    logger.info("Starting Whisper Transcriber (macOS)")
    root = tk.Tk()
    WhisperTranscriber(root)
    root.mainloop()
    logger.info("Closed")

if __name__ == "__main__":
    main()
