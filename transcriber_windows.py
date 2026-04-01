"""
Whisper Transcriber — Windows
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

# ── Logging ───────────────────────────────────────────────────────────────────

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

        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        cmd = [
            ffmpeg_exe, "-nostdin", "-threads", "0",
            "-i", file, "-f", "s16le", "-ac", "1",
            "-acodec", "pcm_s16le", "-ar", str(sr), "-"
        ]
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo
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

# ── Palette ───────────────────────────────────────────────────────────────────

# Neutral light theme — clean, works well on Windows
C_BG            = "#F3F3F3"   # window background (near-system)
C_SURFACE       = "#FFFFFF"   # card / panel surface
C_SURFACE_ALT   = "#FAFAFA"   # transcript area
C_DROP          = "#F7F7F7"   # drop zone fill
C_TEXT          = "#1A1A1A"   # primary text
C_SECONDARY     = "#5A5A5A"   # secondary labels
C_TERTIARY      = "#9A9A9A"   # hints, meta
C_BORDER        = "#DCDCDC"   # borders / dividers
C_BORDER_FOCUS  = "#0078D4"   # Windows blue on focus
C_ACCENT        = "#0078D4"   # Windows 11 accent blue
C_ACCENT_HOVER  = "#106EBE"   # slightly darker
C_ACCENT_FG     = "#FFFFFF"   # text on accent
C_SUCCESS       = "#107C10"   # green
C_WARNING       = "#CA5010"   # orange
C_DANGER        = "#C42B1C"   # red

# ── Typography ────────────────────────────────────────────────────────────────

UI_FONT         = "Segoe UI"
MONO_FONT       = "Consolas"

FONT_TITLE      = (UI_FONT, 16, "bold")
FONT_SECTION    = (UI_FONT, 10, "bold")
FONT_UI         = (UI_FONT, 10)
FONT_UI_MED     = (UI_FONT, 10, "bold")
FONT_SMALL      = (UI_FONT, 9)
FONT_TINY       = (UI_FONT, 8)
FONT_MONO       = (MONO_FONT, 10)
FONT_MONO_SM    = (MONO_FONT, 9)
FONT_BTN        = (UI_FONT, 10)
FONT_BTN_PRI    = (UI_FONT, 10, "bold")

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
    m, s = int(seconds // 60), int(seconds % 60)
    return f"{m:02d}:{s:02d}"

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
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        result = subprocess.run(
            [ffmpeg_exe, "-i", filepath, "-hide_banner"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo,
            timeout=20
        )
        for line in (result.stderr or b"").decode("utf-8", errors="ignore").split('\n'):
            if 'Duration:' in line:
                t = line.split('Duration:')[1].split(',')[0].strip()
                h, m, s = t.split(':')
                return float(h) * 3600 + float(m) * 60 + float(s)
    except Exception as e:
        logger.warning(f"Duration check failed: {e}")
    return None


# ── Flat button widget ────────────────────────────────────────────────────────

class FlatButton(tk.Label):
    """A flat, hover-aware button consistent with the design system."""

    def __init__(self, parent, text, command,
                 bg=C_SURFACE, fg=C_TEXT,
                 hover_bg=C_BORDER,
                 active_bg=C_ACCENT, active_fg=C_ACCENT_FG,
                 font=FONT_BTN,
                 padx=14, pady=5,
                 primary=False,
                 **kwargs):
        self._normal_bg  = C_ACCENT      if primary else bg
        self._normal_fg  = C_ACCENT_FG   if primary else fg
        self._hover_bg   = C_ACCENT_HOVER if primary else hover_bg
        self._hover_fg   = C_ACCENT_FG   if primary else fg
        self._command    = command
        self._disabled   = False

        super().__init__(
            parent,
            text=text,
            font=font,
            bg=self._normal_bg,
            fg=self._normal_fg,
            padx=padx,
            pady=pady,
            cursor="hand2",
            **kwargs
        )
        self.bind("<Enter>",    self._on_enter)
        self.bind("<Leave>",    self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _on_enter(self, e=None):
        if not self._disabled:
            self.config(bg=self._hover_bg, fg=self._hover_fg)

    def _on_leave(self, e=None):
        if not self._disabled:
            self.config(bg=self._normal_bg, fg=self._normal_fg)

    def _on_click(self, e=None):
        if not self._disabled and self._command:
            self._command()

    def set_disabled(self, disabled: bool):
        self._disabled = disabled
        if disabled:
            self.config(bg=C_BORDER, fg=C_TERTIARY, cursor="")
        else:
            self.config(bg=self._normal_bg, fg=self._normal_fg, cursor="hand2")


# ── Tooltip ───────────────────────────────────────────────────────────────────

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
        outer = tk.Frame(tw, bg="#404040", padx=1, pady=1)
        outer.pack()
        tk.Label(
            outer,
            text=self.text,
            justify="left",
            bg="#2D2D2D",
            fg="#F0F0F0",
            font=(UI_FONT, 9),
            padx=9, pady=6,
            wraplength=300
        ).pack()

    def _cancel(self, e=None):
        if self._id:
            self.widget.after_cancel(self._id)
            self._id = None
        if self.tip:
            self.tip.destroy()
            self.tip = None


# ── Main Application ──────────────────────────────────────────────────────────

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
        self._setup_styles()

        if not self._check_deps():
            self.root.after(0, self.root.destroy)
            return

        self._build_ui()
        logger.info("App ready")

    # ── Window ────────────────────────────────────────────────────────────────

    def _setup_window(self):
        self.root.title("Whisper Transcriber")
        self.root.geometry("760x700")
        self.root.minsize(640, 560)
        self.root.configure(bg=C_BG)
        self.root.resizable(True, True)

        self.root.update_idletasks()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"+{(sw - 760) // 2}+{(sh - 700) // 2}")

        # Keyboard shortcuts
        self.root.bind_all("<Control-o>",       lambda e: self._browse_file())
        self.root.bind_all("<Control-s>",       lambda e: self._save_transcript())
        self.root.bind_all("<Control-Shift-c>", lambda e: self._copy_transcript())
        self.root.bind_all("<Control-Shift-C>", lambda e: self._copy_transcript())
        self.root.bind_all("<Control-Return>",  lambda e: self._transcribe())

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            try:
                style.theme_use("winnative")
            except tk.TclError:
                pass

        style.configure(
            "Thin.Horizontal.TProgressbar",
            thickness=3,
            troughcolor=C_BORDER,
            background=C_ACCENT
        )
        style.configure("TCombobox", padding=4)
        style.configure(
            "TScrollbar",
            troughcolor=C_BG,
            background=C_BORDER
        )

    # ── Dependencies ─────────────────────────────────────────────────────────

    def _check_deps(self):
        errors = []
        if not ffmpeg_exe:
            errors.append("imageio-ffmpeg not found")
        if whisper is None:
            errors.append("openai-whisper not found")
        if errors:
            msg = "Missing dependencies:\n\n" + "\n".join(f"  • {e}" for e in errors)
            msg += "\n\nInstall with:\n  pip install openai-whisper imageio-ffmpeg"
            messagebox.showerror("Cannot Start", msg)
            return False
        return True

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Main scroll container
        outer = tk.Frame(self.root, bg=C_BG)
        outer.pack(fill="both", expand=True, padx=24, pady=(18, 0))

        self._build_header(outer)
        self._divider(outer, pady=(12, 0))
        self._build_settings_strip(outer)
        self._build_file_zone(outer)
        self._build_action_row(outer)
        self._build_transcript(outer)
        self._build_statusbar()

    def _divider(self, parent, pady=8):
        tk.Frame(parent, bg=C_BORDER, height=1).pack(fill="x", pady=pady)

    # ── Header ────────────────────────────────────────────────────────────────

    def _build_header(self, parent):
        row = tk.Frame(parent, bg=C_BG)
        row.pack(fill="x")

        tk.Label(row, text="Whisper Transcriber",
                 font=FONT_TITLE, fg=C_TEXT, bg=C_BG, anchor="w").pack(side="left")

        tk.Label(row, text="Local  ·  Private  ·  No cloud",
                 font=FONT_TINY, fg=C_TERTIARY, bg=C_BG).pack(side="right", anchor="e", pady=(4, 0))

    # ── Settings strip ────────────────────────────────────────────────────────

    def _build_settings_strip(self, parent):
        strip = tk.Frame(parent, bg=C_BG)
        strip.pack(fill="x", pady=(14, 0))

        # Model group
        self._build_model_group(strip)
        self._gap(strip, 24)

        # Language
        lang_grp = tk.Frame(strip, bg=C_BG)
        lang_grp.pack(side="left")

        tk.Label(lang_grp, text="Language", font=FONT_SMALL,
                 fg=C_SECONDARY, bg=C_BG).pack(anchor="w")

        self.lang_var = tk.StringVar(value="Auto-detect")
        lc = ttk.Combobox(lang_grp, textvariable=self.lang_var,
                          values=list(LANGUAGES.keys()), width=14, state="readonly")
        lc.pack(anchor="w", pady=(3, 0))
        ToolTip(lc, "Auto-detect works well for most audio.\nSet manually for better accuracy.")

        self._gap(strip, 24)

        # Output mode
        out_grp = tk.Frame(strip, bg=C_BG)
        out_grp.pack(side="left")

        tk.Label(out_grp, text="Output", font=FONT_SMALL,
                 fg=C_SECONDARY, bg=C_BG).pack(anchor="w")

        self.output_mode_var = tk.StringVar(value="Transcribe (original language)")
        oc = ttk.Combobox(out_grp, textvariable=self.output_mode_var,
                          values=list(OUTPUT_MODES.keys()), width=28, state="readonly")
        oc.pack(anchor="w", pady=(3, 0))
        ToolTip(oc, "Transcribe keeps the original language.\nTranslate outputs English text.")

        self._gap(strip, 24)

        # Timestamps
        ts_grp = tk.Frame(strip, bg=C_BG)
        ts_grp.pack(side="left")

        tk.Label(ts_grp, text="Format", font=FONT_SMALL,
                 fg=C_SECONDARY, bg=C_BG).pack(anchor="w")

        self.timestamps_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ts_grp, text="Timestamps",
                        variable=self.timestamps_var).pack(anchor="w", pady=(5, 0))

    def _build_model_group(self, parent):
        grp = tk.Frame(parent, bg=C_BG)
        grp.pack(side="left")

        tk.Label(grp, text="Model", font=FONT_SMALL,
                 fg=C_SECONDARY, bg=C_BG).pack(anchor="w")

        ctrl_row = tk.Frame(grp, bg=C_BG)
        ctrl_row.pack(anchor="w", pady=(3, 0))

        self.model_var = tk.StringVar(value="base")
        self.model_combo = ttk.Combobox(ctrl_row, textvariable=self.model_var,
                                        values=list(MODEL_INFO.keys()),
                                        width=9, state="readonly")
        self.model_combo.pack(side="left")
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        self.load_btn = FlatButton(ctrl_row, text="Load",
                                   command=self._load_model,
                                   primary=True, padx=12, pady=4,
                                   font=FONT_BTN)
        self.load_btn.pack(side="left", padx=(6, 0))
        ToolTip(self.load_btn, "Load model into memory.\nFirst run downloads the model file once.")

        # Status dot + label
        self._dot = tk.Label(ctrl_row, text="●", font=(UI_FONT, 10),
                             fg=C_BORDER, bg=C_BG)
        self._dot.pack(side="left", padx=(10, 0))

        self._model_status = tk.Label(ctrl_row, text="Not loaded",
                                      font=FONT_SMALL, fg=C_SECONDARY, bg=C_BG)
        self._model_status.pack(side="left", padx=(3, 0))

        # Descriptor
        info = MODEL_INFO["base"]
        self._model_desc = tk.Label(
            grp,
            text=f"{info['size']} · {info['speed']} realtime · {info['quality']} — {info['desc']}",
            font=FONT_TINY, fg=C_TERTIARY, bg=C_BG, anchor="w"
        )
        self._model_desc.pack(anchor="w", pady=(3, 0))

    def _gap(self, parent, width=16):
        tk.Frame(parent, bg=C_BG, width=width).pack(side="left")

    def _on_model_change(self, e=None):
        name = self.model_var.get()
        info = MODEL_INFO.get(name, {})
        self._model_desc.config(
            text=f"{info['size']} · {info['speed']} realtime · {info['quality']} — {info['desc']}"
        )

    # ── File zone ─────────────────────────────────────────────────────────────

    def _build_file_zone(self, parent):
        self._file_zone_outer = tk.Frame(parent, bg=C_BG)
        self._file_zone_outer.pack(fill="x", pady=(16, 0))

        # Drop target canvas
        self._drop_canvas = tk.Canvas(
            self._file_zone_outer,
            height=76,
            bg=C_DROP,
            highlightthickness=1,
            highlightbackground=C_BORDER,
            relief="flat",
            cursor="hand2"
        )
        self._drop_canvas.pack(fill="x")
        self._drop_canvas.bind("<Configure>", self._draw_drop_zone)
        self._drop_canvas.bind("<Button-1>",  lambda e: self._browse_file())
        self._drop_canvas.bind("<Enter>",     self._drop_hover_on)
        self._drop_canvas.bind("<Leave>",     self._drop_hover_off)

        # File info strip (shown after a file is selected)
        self._file_strip = tk.Frame(self._file_zone_outer, bg=C_BG)

        strip_inner = tk.Frame(self._file_strip, bg=C_BG)
        strip_inner.pack(fill="x", pady=(10, 0))

        self._file_icon_lbl = tk.Label(strip_inner, text="♫",
                                       font=(UI_FONT, 15), fg=C_ACCENT, bg=C_BG)
        self._file_icon_lbl.pack(side="left")

        text_col = tk.Frame(strip_inner, bg=C_BG)
        text_col.pack(side="left", padx=(8, 0), fill="x", expand=True)

        self._file_name_lbl = tk.Label(text_col, text="",
                                       font=FONT_UI_MED, fg=C_TEXT, bg=C_BG, anchor="w")
        self._file_name_lbl.pack(fill="x")

        self._file_meta_lbl = tk.Label(text_col, text="",
                                       font=FONT_TINY, fg=C_SECONDARY, bg=C_BG, anchor="w")
        self._file_meta_lbl.pack(fill="x")

        FlatButton(strip_inner, text="Change…",
                   command=self._browse_file,
                   padx=10, pady=3, font=FONT_SMALL
                   ).pack(side="right")

        self.file_path = tk.StringVar()

        # Drag-and-drop (tkinterdnd2 optional)
        try:
            self.root.drop_target_register("DND_Files")
            self.root.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            pass

    def _draw_drop_zone(self, event=None):
        c = self._drop_canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        cx, cy = w // 2, h // 2

        pad = 6
        c.create_rectangle(pad, pad, w - pad, h - pad,
                           outline=C_BORDER, dash=(5, 4), width=1)

        c.create_text(cx, cy - 8, text="↓",
                      font=(UI_FONT, 18), fill=C_TERTIARY, anchor="center")
        c.create_text(cx, cy + 14, text="Drop a file here, or click to browse",
                      font=(UI_FONT, 10), fill=C_SECONDARY, anchor="center")
        c.create_text(cx, cy + 30, text="mp3 · wav · m4a · mp4 · mov · and more",
                      font=(UI_FONT, 8), fill=C_TERTIARY, anchor="center")

    def _drop_hover_on(self, e=None):
        self._drop_canvas.configure(bg="#EBF3FB",
                                    highlightbackground=C_BORDER_FOCUS)

    def _drop_hover_off(self, e=None):
        self._drop_canvas.configure(bg=C_DROP,
                                    highlightbackground=C_BORDER)

    def _on_drop(self, event):
        if self.is_transcribing or self.is_loading_model:
            self._set_status("Busy — please wait")
            return
        try:
            paths = self.root.tk.splitlist(event.data)
        except Exception:
            paths = [event.data]
        if not paths:
            return
        self._set_file(paths[0].strip().strip("{}"))

    def _show_file_strip(self, name, meta):
        ext = Path(name).suffix.lower()
        icon = "▶" if ext in (".mp4", ".webm", ".mkv", ".avi", ".mov") else "♫"
        self._file_icon_lbl.config(text=icon)
        self._file_name_lbl.config(text=name)
        self._file_meta_lbl.config(text=meta)
        self._drop_canvas.pack_forget()
        self._file_strip.pack(fill="x")

    def _reset_file_zone(self):
        self._file_strip.pack_forget()
        self._drop_canvas.config(height=76, bg=C_DROP, highlightbackground=C_BORDER)
        self._drop_canvas.pack(fill="x")

    # ── Action row ────────────────────────────────────────────────────────────

    def _build_action_row(self, parent):
        row = tk.Frame(parent, bg=C_BG)
        row.pack(fill="x", pady=(14, 0))

        self.transcribe_btn = FlatButton(
            row, text="Transcribe",
            command=self._transcribe,
            primary=True, padx=22, pady=7,
            font=FONT_BTN_PRI
        )
        self.transcribe_btn.pack(side="left")
        ToolTip(self.transcribe_btn, "Start transcription  Ctrl+Enter")

        self.cancel_btn = FlatButton(
            row, text="Cancel",
            command=self._cancel,
            bg=C_SURFACE, padx=14, pady=7,
            font=FONT_BTN
        )
        # hidden until transcribing

        self.progress = ttk.Progressbar(
            row, mode="indeterminate", length=100,
            style="Thin.Horizontal.TProgressbar"
        )
        # hidden until transcribing

        self._elapsed_lbl = tk.Label(row, text="", font=FONT_MONO_SM,
                                     fg=C_SECONDARY, bg=C_BG)
        # hidden until transcribing

        self._action_info = tk.Label(row, text="", font=FONT_SMALL,
                                     fg=C_SECONDARY, bg=C_BG)
        # hidden until transcribing

    # ── Transcript ────────────────────────────────────────────────────────────

    def _build_transcript(self, parent):
        # Toolbar
        bar = tk.Frame(parent, bg=C_BG)
        bar.pack(fill="x", pady=(18, 6))

        tk.Label(bar, text="Transcript", font=FONT_SECTION,
                 fg=C_TEXT, bg=C_BG).pack(side="left")

        self._word_count = tk.Label(bar, text="", font=FONT_TINY,
                                    fg=C_TERTIARY, bg=C_BG)
        self._word_count.pack(side="right")

        FlatButton(bar, text="Clear", command=self._clear_output,
                   padx=10, pady=3, font=FONT_SMALL
                   ).pack(side="right", padx=(4, 8))
        FlatButton(bar, text="Copy", command=self._copy_transcript,
                   padx=10, pady=3, font=FONT_SMALL
                   ).pack(side="right", padx=(4, 0))
        FlatButton(bar, text="Save…", command=self._save_transcript,
                   padx=10, pady=3, font=FONT_SMALL
                   ).pack(side="right", padx=(4, 0))

        # Text area with border
        border = tk.Frame(parent, bg=C_BORDER)
        border.pack(fill="both", expand=True)
        inner = tk.Frame(border, bg=C_SURFACE_ALT)
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        self.output = tk.Text(
            inner,
            font=FONT_MONO,
            wrap=tk.WORD,
            relief="flat",
            borderwidth=0,
            bg=C_SURFACE_ALT,
            fg=C_TEXT,
            insertbackground=C_TEXT,
            selectbackground=C_ACCENT,
            selectforeground=C_ACCENT_FG,
            padx=14,
            pady=12,
            spacing2=3,
            undo=True
        )
        scrollbar = ttk.Scrollbar(inner, orient="vertical", command=self.output.yview)
        self.output.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
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
        bar = tk.Frame(self.root, bg=C_BG)
        bar.pack(fill="x", side="bottom")

        self._status_var = tk.StringVar(value="Ready")
        tk.Label(bar, textvariable=self._status_var,
                 font=FONT_TINY, fg=C_SECONDARY, bg=C_BG, anchor="w"
                 ).pack(side="left", padx=14, pady=(4, 5))

        tk.Label(bar, text="Local only · No cloud",
                 font=FONT_TINY, fg=C_TERTIARY, bg=C_BG
                 ).pack(side="right", padx=14, pady=(4, 5))

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_text_modified(self, e=None):
        if self.output.edit_modified():
            if not getattr(self, '_placeholder_active', False):
                text = self.output.get("1.0", tk.END).strip()
                self._word_count.config(
                    text=f"{len(text.split()):,} words" if text else ""
                )
            self.output.edit_modified(False)

    # ── Load Model ────────────────────────────────────────────────────────────

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
                self._set_status(f"{name.capitalize()} loaded ({elapsed:.1f}s)")
            except Exception as e:
                logger.error(f"Load failed: {e}")
                self._run_on_ui(lambda: self._set_dot(C_DANGER, "Load failed"))
                self._run_on_ui(lambda: messagebox.showerror("Model Error", f"Could not load model:\n\n{e}"))
                self._set_status("Error loading model")
            finally:
                self.is_loading_model = False
                self._set_busy(False)

        threading.Thread(target=task, daemon=True).start()

    # ── Browse / Drop ─────────────────────────────────────────────────────────

    def _browse_file(self):
        if self.is_transcribing or self.is_loading_model:
            return
        path = filedialog.askopenfilename(
            title="Open Audio or Video File",
            filetypes=[
                ("Audio/Video", " ".join(f"*{e}" for e in SUPPORTED_FORMATS)),
                ("Audio",       "*.mp3 *.wav *.m4a *.ogg *.flac"),
                ("Video",       "*.mp4 *.webm *.mkv *.avi *.mov"),
                ("All files",   "*.*"),
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
        if self.is_transcribing:
            return
        if self.is_loading_model:
            messagebox.showinfo("Busy", "Model is still loading.")
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
        with_timestamps  = self.timestamps_var.get()
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

                text = self._format_timestamps(result) if with_timestamps else result["text"].strip()
                detected = result.get("language", "?")
                self._run_on_ui(lambda: self._show_result(text, elapsed, detected, output_mode_disp))

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
        self._run_on_ui(self._prepare_live_output)

        audio = whisper.audio.load_audio(filepath)
        sr = whisper.audio.SAMPLE_RATE
        total_s = len(audio) / sr if len(audio) else 0
        n_chunks = max(1, math.ceil(len(audio) / (sr * 30)))

        segments, chunk_texts, detected_lang = [], [], language

        for i in range(n_chunks):
            if not self.is_transcribing:
                break

            s0 = i * sr * 30
            s1 = min(len(audio), s0 + sr * 30)
            chunk = audio[s0:s1]
            if not len(chunk):
                continue

            res = self.model.transcribe(chunk, language=detected_lang,
                                        task=task_name, verbose=False)
            if not detected_lang:
                detected_lang = res.get("language")

            offset = s0 / sr
            for seg in res.get("segments", []):
                adj = dict(seg)
                adj["start"] = seg["start"] + offset
                adj["end"]   = seg["end"]   + offset
                segments.append(adj)

            t = res.get("text", "").strip()
            if t:
                chunk_texts.append(t)

            processed = min(s1 / sr, total_s)
            preview = self._build_preview(chunk_texts, segments, with_timestamps)
            self._run_on_ui(
                lambda txt=preview, p=processed, tot=total_s:
                    self._live_update(txt, p, tot)
            )

        return {"text": " ".join(chunk_texts).strip(),
                "segments": segments,
                "language": detected_lang or "?"}

    def _build_preview(self, chunk_texts, segments, with_timestamps):
        if with_timestamps:
            return "\n\n".join(
                f"[{format_duration(s['start'])} → {format_duration(s['end'])}]  {s['text'].strip()}"
                for s in segments if s.get("text", "").strip()
            )
        return "\n\n".join(chunk_texts)

    def _prepare_live_output(self):
        self._clear_placeholder()
        self.output.delete("1.0", tk.END)

    def _live_update(self, text, processed, total):
        self._clear_placeholder()
        self.output.delete("1.0", tk.END)
        if text:
            self.output.insert("1.0", text)
            self.output.see(tk.END)
        self.output.edit_modified(True)

        total_ = total or processed or 1
        pct = max(0, min(100, int(processed / total_ * 100))) if total_ else 0
        self.progress.configure(mode="determinate", maximum=100, value=pct)
        self._action_info.config(
            text=f"Transcribing  {pct}%  "
                 f"({format_duration(processed)} / {format_duration(total) if total else '?'})"
        )
        self._set_status(
            f"Transcribing  {pct}%  "
            f"({format_duration(processed)} / {format_duration(total) if total else '?'})"
        )

    def _format_timestamps(self, result):
        return "\n\n".join(
            f"[{format_duration(s['start'])} → {format_duration(s['end'])}]  {s['text'].strip()}"
            for s in result.get("segments", [])
        )

    def _show_result(self, text, elapsed, lang, mode_disp):
        self._clear_placeholder()
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)
        label = "Translated" if "Translate" in mode_disp else "Transcribed"
        self._set_status(f"{label} in {format_duration(elapsed)} · Language: {lang}")

    def _cancel(self):
        if self.is_transcribing:
            self.is_transcribing = False
            self._set_status("Cancelled")

    # ── Transcription UI state ────────────────────────────────────────────────

    def _show_transcription_ui(self):
        self.cancel_btn.pack(side="left", padx=(8, 0))
        self.progress.configure(mode="determinate", maximum=100, value=0)
        self.progress.pack(side="left", padx=(12, 0))
        self._elapsed_lbl.config(text="00:00")
        self._elapsed_lbl.pack(side="left", padx=(10, 0))
        self._action_info.config(text="Preparing…")
        self._action_info.pack(side="left", padx=(10, 0))
        self._start_elapsed_timer()

    def _hide_transcription_ui(self):
        self._stop_elapsed_timer()
        self.cancel_btn.pack_forget()
        self.progress.pack_forget()
        self._elapsed_lbl.pack_forget()
        self._action_info.pack_forget()

    def _start_elapsed_timer(self):
        self._stop_elapsed_timer()
        def tick():
            if self.is_transcribing and self._transcription_start:
                self._elapsed_lbl.config(
                    text=format_clock(time.time() - self._transcription_start)
                )
                self._elapsed_timer_id = self.root.after(1000, tick)
        tick()

    def _stop_elapsed_timer(self):
        if self._elapsed_timer_id:
            self.root.after_cancel(self._elapsed_timer_id)
            self._elapsed_timer_id = None

    # ── Transcript actions ────────────────────────────────────────────────────

    def _save_transcript(self):
        text = self._transcript_text()
        if not text:
            messagebox.showwarning("Nothing to Save", "Transcript is empty.")
            return
        default = "transcript.txt"
        if self.file_path.get():
            stem = Path(self.file_path.get()).stem
            suffix = "_english_translation" if OUTPUT_MODES.get(self.output_mode_var.get()) == "en" else "_transcript"
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
        text = self._transcript_text()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status("Copied to clipboard")

    def _clear_output(self):
        if not self._transcript_text():
            return
        self._show_placeholder()
        self._word_count.config(text="")
        self._set_status("Cleared")

    def _transcript_text(self):
        if getattr(self, '_placeholder_active', False):
            return ""
        return self.output.get("1.0", tk.END).strip()

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _set_busy(self, busy, msg=None):
        def apply():
            if busy:
                self.transcribe_btn.set_disabled(True)
                self.load_btn.set_disabled(True)
            else:
                self.progress.configure(mode="determinate", maximum=100, value=0)
                self.transcribe_btn.set_disabled(False)
                self.load_btn.set_disabled(False)
        self._run_on_ui(apply)
        if msg:
            self._set_status(msg)

    def _set_status(self, msg):
        def apply():
            if hasattr(self, "_status_var"):
                self._status_var.set(msg)
        self._run_on_ui(apply)
        logger.info(f"Status: {msg}")

    def _set_dot(self, color, text):
        def apply():
            if hasattr(self, "_dot"):
                self._dot.config(fg=color)
                self._model_status.config(text=text)
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
        self._stop_elapsed_timer()
        try:
            self.root.destroy()
        except tk.TclError:
            pass


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    logger.info("Starting Whisper Transcriber (Windows)")
    root = tk.Tk()
    WhisperTranscriber(root)
    root.mainloop()
    logger.info("Closed")

if __name__ == "__main__":
    main()
