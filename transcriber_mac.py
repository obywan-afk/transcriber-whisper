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
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
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

# ── Constants ─────────────────────────────────────────────────────────────────

SYS_BG          = "systemWindowBackgroundColor"
SYS_TEXT        = "systemTextColor"
SYS_SECONDARY   = "systemSecondaryLabelColor"
SYS_TERTIARY    = "systemTertiaryLabelColor"
SYS_SEPARATOR   = "systemSeparatorColor"
SYS_CTRL_BG     = "systemControlBackgroundColor"
SYS_BLUE        = "systemBlueColor"

HEX_BG          = "#ECECEC"
HEX_CARD        = "#FFFFFF"
HEX_TEXT        = "#1D1D1F"
HEX_SECONDARY   = "#6E6E73"
HEX_TERTIARY    = "#AEAEB2"
HEX_SEPARATOR   = "#D1D1D6"
HEX_BLUE        = "#007AFF"
HEX_RED_DOT     = "#FF3B30"
HEX_GREEN       = "#30D158"
HEX_ORANGE      = "#FF9500"
HEX_DROP_BG     = "#F5F5F7"
HEX_DROP_BORDER = "#C7C7CC"
HEX_DROP_ACTIVE = "#E8F0FE"

FONT_TITLE      = ("Helvetica Neue", 20, "bold")
FONT_SECTION    = ("Helvetica Neue", 13, "bold")
FONT_UI         = ("Helvetica Neue", 13)
FONT_UI_MED     = ("Helvetica Neue", 13, "bold")
FONT_SMALL      = ("Helvetica Neue", 11)
FONT_TINY       = ("Helvetica Neue", 10)
FONT_LABEL      = ("Helvetica Neue", 11)
FONT_MONO       = ("Menlo", 12)
FONT_STATUS     = ("Helvetica Neue", 11)
FONT_DROP_ICON  = ("Helvetica Neue", 28)
FONT_DROP_TEXT  = ("Helvetica Neue", 12)
FONT_DROP_HINT  = ("Helvetica Neue", 11)
FONT_MODEL_NAME = ("Helvetica Neue", 12, "bold")
FONT_MODEL_DESC = ("Helvetica Neue", 10)
FONT_ELAPSED    = ("Menlo", 11)

SUPPORTED_FORMATS = (
    ".mp3", ".wav", ".m4a", ".ogg", ".flac",
    ".mp4", ".webm", ".mkv", ".avi", ".mov"
)

MODEL_INFO = {
    "tiny":   {"size": "39 MB",   "speed": "~1x",  "accuracy": "Basic",  "desc": "Quick drafts"},
    "base":   {"size": "74 MB",   "speed": "~3x",  "accuracy": "Good",   "desc": "Everyday use"},
    "small":  {"size": "244 MB",  "speed": "~6x",  "accuracy": "Better", "desc": "Accurate"},
    "medium": {"size": "769 MB",  "speed": "~18x", "accuracy": "Great",  "desc": "High accuracy"},
    "large":  {"size": "1.5 GB",  "speed": "~32x", "accuracy": "Best",   "desc": "Maximum quality"},
}

LANGUAGES = {
    "Auto-detect": None,
    "English": "en", "Finnish": "fi", "Swedish": "sv",
    "German": "de",  "French": "fr",  "Spanish": "es",
    "Italian": "it", "Dutch": "nl",   "Polish": "pl",
    "Portuguese": "pt", "Russian": "ru",
    "Mandarin Chinese (Modern Standard)": "zh",
    "Japanese": "ja", "Korean": "ko",
}

OUTPUT_LANGUAGES = {
    "Original language (transcribe)": "source",
    "English (translate)": "en",
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
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"

def format_elapsed_clock(seconds):
    """Format as mm:ss for the live timer."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def get_file_duration(filepath):
    if not ffmpeg_exe:
        return None
    try:
        result = subprocess.run(
            [ffmpeg_exe, "-i", filepath, "-hide_banner"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20
        )
        stderr_text = (result.stderr or b"").decode("utf-8", errors="ignore")
        for line in stderr_text.split('\n'):
            if 'Duration:' in line:
                t = line.split('Duration:')[1].split(',')[0].strip()
                h, m, s = t.split(':')
                return float(h) * 3600 + float(m) * 60 + float(s)
    except subprocess.TimeoutExpired:
        logger.warning("Duration check timed out")
    except Exception as e:
        logger.warning(f"Duration check failed: {e}")
    return None

def format_file_size(size_bytes):
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


# ── Tooltip ───────────────────────────────────────────────────────────────────

class ToolTip:
    def __init__(self, widget, text, delay=600):
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
        outer = tk.Frame(tw, bg="#1C1C1E", padx=1, pady=1)
        outer.pack()
        inner = tk.Frame(outer, bg="#2C2C2E")
        inner.pack()
        tk.Label(
            inner, text=self.text, justify="left",
            bg="#2C2C2E", fg="#EBEBF5",
            font=("Helvetica Neue", 11),
            padx=9, pady=6, wraplength=280
        ).pack()

    def _cancel(self, e=None):
        if self._id:
            self.widget.after_cancel(self._id)
            self._id = None
        if self.tip:
            self.tip.destroy()
            self.tip = None


# ── Main App ──────────────────────────────────────────────────────────────────

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
        self._build_menubar()

        if not self._check_deps():
            self.root.after(0, self.root.destroy)
            return

        self._build_ui()
        logger.info("App ready")

    # ── Window & Style ────────────────────────────────────────────────────────

    def _setup_window(self):
        self.root.title("Whisper Transcriber")
        self.root.geometry("760x720")
        self.root.minsize(640, 580)
        self.root.resizable(True, True)

        self.root.update_idletasks()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"+{(sw - 760) // 2}+{(sh - 720) // 2}")

        try:
            self.root.configure(bg=SYS_BG)
        except tk.TclError:
            self.root.configure(bg=HEX_BG)

        self.color_bg = self._resolve(SYS_BG, HEX_BG)
        self.color_text = self._resolve(SYS_TEXT, HEX_TEXT)
        self.color_secondary = self._resolve(SYS_SECONDARY, HEX_SECONDARY)
        self.color_tertiary = self._resolve(SYS_TERTIARY, HEX_TERTIARY)
        self.color_separator = self._resolve(SYS_SEPARATOR, HEX_SEPARATOR)
        self.color_text_bg = self._resolve("systemTextBackgroundColor", HEX_CARD)
        self.color_blue = self._resolve(SYS_BLUE, HEX_BLUE)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _resolve(self, token, fallback):
        try:
            self.root.winfo_rgb(token)
            return token
        except tk.TclError:
            return fallback

    def _setup_styles(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("aqua")
        except tk.TclError:
            pass
        self.style.configure("Thin.Horizontal.TProgressbar", thickness=3)

    def _build_menubar(self):
        menubar = tk.Menu(self.root)

        app_menu = tk.Menu(menubar, name="apple", tearoff=False)
        menubar.add_cascade(menu=app_menu)
        app_menu.add_command(label="About Whisper Transcriber", command=self._show_about)

        file_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open…", accelerator="⌘O", command=self._browse_file)
        file_menu.add_separator()
        file_menu.add_command(label="Save Transcript…", accelerator="⌘S", command=self._save_transcript)
        file_menu.add_separator()
        file_menu.add_command(label="Close Window", accelerator="⌘W", command=self._on_close)

        edit_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Copy Transcript", accelerator="⌘⇧C", command=self._copy_transcript)
        edit_menu.add_command(label="Clear Transcript", command=self._clear_output)

        help_menu = tk.Menu(menubar, name="help", tearoff=False)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Whisper Transcriber Help", command=self._show_help)

        self.root.configure(menu=menubar)

        self.root.bind_all("<Command-o>", lambda e: self._browse_file())
        self.root.bind_all("<Command-s>", lambda e: self._save_transcript())
        self.root.bind_all("<Command-Shift-c>", lambda e: self._copy_transcript())
        self.root.bind_all("<Command-Shift-C>", lambda e: self._copy_transcript())
        self.root.bind_all("<Command-w>", lambda e: self._on_close())
        self.root.bind_all("<Command-Return>", lambda e: self._transcribe())

        try:
            self.root.createcommand("tk::mac::Quit", self._on_close)
        except Exception:
            pass

    # ── Dependency check ──────────────────────────────────────────────────────

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
            logger.error(f"Deps missing: {errors}")
            return False
        return True

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        bg = self._bg()

        outer = tk.Frame(self.root, bg=bg)
        outer.pack(fill="both", expand=True, padx=28, pady=(12, 0))

        # ── Header ──
        header = tk.Frame(outer, bg=bg)
        header.pack(fill="x", pady=(0, 6))

        tk.Label(header, text="Whisper Transcriber", font=FONT_TITLE,
                 fg=self.color_text, bg=bg, anchor="w").pack(side="left")

        tk.Label(header, text="Local-only transcription · OpenAI Whisper", font=FONT_SMALL,
                 fg=self.color_tertiary, bg=bg, anchor="e").pack(side="right", pady=(6, 0))

        # ── Top controls: Model + Language + Timestamps in one row ──
        self._build_controls_row(outer)

        # ── File drop zone ──
        self._build_drop_zone(outer)

        # ── Action row ──
        self._build_action_row(outer)

        # ── Transcript ──
        self._build_transcript_section(outer)

        # ── Status bar ──
        self._build_statusbar()

    def _bg(self):
        return self.color_bg

    def _sep(self, parent, pady=10):
        try:
            ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=pady)
        except Exception:
            tk.Frame(parent, bg=HEX_SEPARATOR, height=1).pack(fill="x", pady=pady)

    # ── Controls row (model + language + timestamps) ──────────────────────────

    def _build_controls_row(self, parent):
        bg = self._bg()

        row = tk.Frame(parent, bg=bg)
        row.pack(fill="x", pady=(14, 0))

        # Model
        model_group = tk.Frame(row, bg=bg)
        model_group.pack(side="left")

        tk.Label(model_group, text="Model", font=FONT_SMALL,
                 fg=self.color_secondary, bg=bg).pack(anchor="w")

        model_inner = tk.Frame(model_group, bg=bg)
        model_inner.pack(anchor="w", pady=(3, 0))

        self.model_var = tk.StringVar(value="base")
        self.model_combo = ttk.Combobox(
            model_inner,
            textvariable=self.model_var,
            values=list(MODEL_INFO.keys()),
            width=8,
            state="readonly"
        )
        self.model_combo.pack(side="left")
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        self.load_btn = ttk.Button(
            model_inner, text="Load", command=self._load_model, width=5
        )
        self.load_btn.pack(side="left", padx=(6, 0))

        # Model status dot + text
        self._dot = tk.Label(model_inner, text="●", font=("Helvetica Neue", 9),
                             fg=self.color_separator, bg=bg)
        self._dot.pack(side="left", padx=(10, 0))

        self._model_status = tk.Label(model_inner, text="Not loaded",
                                      font=FONT_TINY, fg=self.color_secondary, bg=bg)
        self._model_status.pack(side="left", padx=(4, 0))

        # Model description below
        self._model_desc = tk.Label(model_group,
                                    text="74 MB · ~3x realtime · Good — Everyday use",
                                    font=FONT_TINY, fg=self.color_tertiary, bg=bg, anchor="w")
        self._model_desc.pack(anchor="w", pady=(2, 0))

        # Spacer
        tk.Frame(row, bg=bg, width=32).pack(side="left")

        # Language
        lang_group = tk.Frame(row, bg=bg)
        lang_group.pack(side="left")

        tk.Label(lang_group, text="Language", font=FONT_SMALL,
                 fg=self.color_secondary, bg=bg).pack(anchor="w")

        self.lang_var = tk.StringVar(value="Auto-detect")
        lang_combo = ttk.Combobox(
            lang_group,
            textvariable=self.lang_var,
            values=list(LANGUAGES.keys()),
            width=13,
            state="readonly"
        )
        lang_combo.pack(anchor="w", pady=(3, 0))
        ToolTip(lang_combo, "Auto-detect works well for most audio.\nSet manually to improve accuracy.")

        # Spacer
        tk.Frame(row, bg=bg, width=32).pack(side="left")

        # Output mode
        output_group = tk.Frame(row, bg=bg)
        output_group.pack(side="left")

        tk.Label(output_group, text="Output", font=FONT_SMALL,
                 fg=self.color_secondary, bg=bg).pack(anchor="w")

        self.output_lang_var = tk.StringVar(value="Original language (transcribe)")
        output_combo = ttk.Combobox(
            output_group,
            textvariable=self.output_lang_var,
            values=list(OUTPUT_LANGUAGES.keys()),
            width=30,
            state="readonly"
        )
        output_combo.pack(anchor="w", pady=(3, 0))
        ToolTip(
            output_combo,
            "Choose output mode:\n"
            "• Original language = normal transcription\n"
            "• English = Whisper built-in translation\n\n"
            "Whisper directly translates speech to English only."
        )

        # Spacer
        tk.Frame(row, bg=bg, width=32).pack(side="left")

        # Timestamps
        ts_group = tk.Frame(row, bg=bg)
        ts_group.pack(side="left")

        tk.Label(ts_group, text="Format", font=FONT_SMALL,
                 fg=self.color_secondary, bg=bg).pack(anchor="w")

        self.timestamps_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            ts_group, text="Timestamps", variable=self.timestamps_var
        ).pack(anchor="w", pady=(5, 0))

    def _on_model_change(self, e=None):
        name = self.model_var.get()
        info = MODEL_INFO.get(name, {})
        self._model_desc.config(
            text=f"{info['size']} · {info['speed']} realtime · {info['accuracy']} — {info['desc']}"
        )

    # ── File drop zone ────────────────────────────────────────────────────────

    def _build_drop_zone(self, parent):
        bg = self._bg()

        self._drop_outer = tk.Frame(parent, bg=bg)
        self._drop_outer.pack(fill="x", pady=(16, 0))

        # The drop zone itself — a rounded-rect-ish area with dashed border feel
        self._drop_zone = tk.Canvas(
            self._drop_outer,
            height=88,
            bg=HEX_DROP_BG,
            highlightthickness=1,
            highlightbackground=HEX_DROP_BORDER,
            relief="flat",
            cursor="hand2"
        )
        self._drop_zone.pack(fill="x")

        # Draw content on canvas
        self._drop_zone.bind("<Configure>", self._draw_drop_content)
        self._drop_zone.bind("<Button-1>", lambda e: self._browse_file())

        # File info row (hidden until file is selected)
        self._file_info_frame = tk.Frame(self._drop_outer, bg=bg)

        file_info_inner = tk.Frame(self._file_info_frame, bg=bg)
        file_info_inner.pack(fill="x", pady=(8, 0))

        self._file_icon = tk.Label(file_info_inner, text="♫", font=("Helvetica Neue", 14),
                                   fg=self.color_blue, bg=bg)
        self._file_icon.pack(side="left")

        file_text_col = tk.Frame(file_info_inner, bg=bg)
        file_text_col.pack(side="left", padx=(6, 0), fill="x", expand=True)

        self._file_name_label = tk.Label(file_text_col, text="", font=FONT_UI,
                                         fg=self.color_text, bg=bg, anchor="w")
        self._file_name_label.pack(fill="x")

        self._file_meta_label = tk.Label(file_text_col, text="", font=FONT_TINY,
                                         fg=self.color_secondary, bg=bg, anchor="w")
        self._file_meta_label.pack(fill="x")

        self._file_change_btn = ttk.Button(file_info_inner, text="Change…",
                                           command=self._browse_file, width=8)
        self._file_change_btn.pack(side="right")

        self.file_path = tk.StringVar()

        # Drag-and-drop support
        try:
            self.root.drop_target_register("DND_Files")
            self.root.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            pass

    def _draw_drop_content(self, event=None):
        c = self._drop_zone
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()

        # Dashed border rectangle (inset)
        inset = 6
        c.create_rectangle(
            inset, inset, w - inset, h - inset,
            outline=HEX_DROP_BORDER, dash=(6, 4), width=1
        )

        # Icon and text
        cx = w // 2
        cy = h // 2 - 6
        c.create_text(cx, cy - 8, text="↓", font=FONT_DROP_ICON,
                      fill=HEX_SECONDARY, anchor="center")
        c.create_text(cx, cy + 24, text="Drop audio or video file here",
                      font=FONT_DROP_TEXT, fill=HEX_SECONDARY, anchor="center")
        c.create_text(cx, cy + 42, text="or click to browse · mp3, wav, m4a, mp4, mov…",
                      font=FONT_DROP_HINT, fill=HEX_TERTIARY, anchor="center")

    def _on_drop(self, event):
        if self.is_transcribing or self.is_loading_model:
            self._set_status("Busy — wait for current task to finish")
            return
        try:
            paths = self.root.tk.splitlist(event.data)
        except Exception:
            paths = [event.data]
        if not paths:
            return
        raw = paths[0].strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        self._set_file(raw)

    def _show_file_info(self, name, meta):
        """Switch from drop zone to file info display."""
        # Determine icon based on extension
        ext = Path(name).suffix.lower()
        if ext in (".mp4", ".webm", ".mkv", ".avi", ".mov"):
            self._file_icon.config(text="🎬")
        else:
            self._file_icon.config(text="♫")

        self._file_name_label.config(text=name)
        self._file_meta_label.config(text=meta)

        # Shrink drop zone, show file info
        self._drop_zone.config(height=0)
        self._drop_zone.pack_forget()
        self._file_info_frame.pack(fill="x")

    def _reset_drop_zone(self):
        """Switch back to drop zone from file info."""
        self._file_info_frame.pack_forget()
        self._drop_zone.config(height=88)
        self._drop_zone.pack(fill="x")

    # ── Action row ────────────────────────────────────────────────────────────

    def _build_action_row(self, parent):
        bg = self._bg()
        row = tk.Frame(parent, bg=bg)
        row.pack(fill="x", pady=(14, 0))

        # Transcribe button — default active gets macOS blue accent
        self.transcribe_btn = ttk.Button(
            row, text="Transcribe", command=self._transcribe, default="active"
        )
        self.transcribe_btn.pack(side="left")

        ToolTip(self.transcribe_btn, "Start transcription (⌘Return)")

        # Cancel button (hidden until transcribing)
        self.cancel_btn = ttk.Button(row, text="Cancel", command=self._cancel)

        # Progress bar
        self.progress = ttk.Progressbar(
            row, mode="indeterminate", length=120,
            style="Thin.Horizontal.TProgressbar"
        )
        # Not packed by default — shown during transcription

        # Elapsed time label
        self._elapsed_label = tk.Label(row, text="", font=FONT_ELAPSED,
                                       fg=self.color_secondary, bg=bg)
        # Not packed by default

        # Transcription status text (shows during transcription)
        self._action_status = tk.Label(row, text="", font=FONT_SMALL,
                                       fg=self.color_secondary, bg=bg)

    # ── Transcript section ────────────────────────────────────────────────────

    def _build_transcript_section(self, parent):
        bg = self._bg()

        # Header row
        toolbar = tk.Frame(parent, bg=bg)
        toolbar.pack(fill="x", pady=(16, 6))

        tk.Label(toolbar, text="Transcript", font=FONT_SECTION,
                 fg=self.color_text, bg=bg).pack(side="left")

        # Word count
        self._word_count = tk.Label(toolbar, text="", font=FONT_TINY,
                                    fg=self.color_tertiary, bg=bg)
        self._word_count.pack(side="right")

        # Transcript action buttons
        self.clear_btn = ttk.Button(toolbar, text="Clear", command=self._clear_output, width=5)
        self.clear_btn.pack(side="right", padx=(4, 8))

        self.copy_btn = ttk.Button(toolbar, text="Copy", command=self._copy_transcript, width=5)
        self.copy_btn.pack(side="right", padx=(4, 0))

        self.save_btn = ttk.Button(toolbar, text="Save…", command=self._save_transcript, width=5)
        self.save_btn.pack(side="right", padx=(4, 0))

        # Text area with border frame
        text_border = tk.Frame(parent, bg=self.color_separator, bd=0)
        text_border.pack(fill="both", expand=True, pady=(0, 0))

        text_inner = tk.Frame(text_border)
        text_inner.pack(fill="both", expand=True, padx=1, pady=1)

        self.output = tk.Text(
            text_inner,
            font=FONT_MONO,
            wrap=tk.WORD,
            relief="flat",
            borderwidth=0,
            bg=self.color_text_bg,
            fg=self.color_text,
            insertbackground=self.color_text,
            selectbackground=self.color_blue,
            selectforeground="white",
            padx=14,
            pady=12,
            spacing2=3,
            undo=True
        )

        scrollbar = ttk.Scrollbar(text_inner, orient="vertical", command=self.output.yview)
        self.output.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.output.pack(side="left", fill="both", expand=True)

        self.output.bind("<<Modified>>", self._on_text_modified)

        # Placeholder / empty state
        self._show_placeholder()

    def _show_placeholder(self):
        """Show subtle placeholder text in the transcript area."""
        self.output.config(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.insert("1.0", "Your transcript will appear here…")
        self.output.config(fg=self.color_tertiary)
        self._placeholder_active = True

    def _clear_placeholder(self):
        if getattr(self, '_placeholder_active', False):
            self.output.config(state="normal")
            self.output.delete("1.0", tk.END)
            self.output.config(fg=self.color_text)
            self._placeholder_active = False

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=self.color_separator, height=1)
        bar.pack(fill="x", side="bottom")

        status_bg = self.color_bg
        statusbar = tk.Frame(self.root, bg=status_bg)
        statusbar.pack(fill="x", side="bottom")

        self._status_var = tk.StringVar(value="Ready")
        tk.Label(
            statusbar,
            textvariable=self._status_var,
            font=FONT_STATUS,
            fg=self.color_secondary,
            bg=status_bg,
            anchor="w"
        ).pack(side="left", padx=14, pady=(4, 6))

        tk.Label(
            statusbar,
            text="Whisper · Local only · No cloud upload",
            font=FONT_TINY,
            fg=self.color_tertiary,
            bg=status_bg
        ).pack(side="right", padx=14, pady=(4, 6))

    # ── Event Handlers ────────────────────────────────────────────────────────

    def _on_text_modified(self, e=None):
        if self.output.edit_modified():
            if not getattr(self, '_placeholder_active', False):
                text = self.output.get("1.0", tk.END).strip()
                if text:
                    words = len(text.split())
                    self._word_count.config(text=f"{words:,} words")
                else:
                    self._word_count.config(text="")
            self.output.edit_modified(False)

    # ── Core: Load Model ──────────────────────────────────────────────────────

    def _load_model(self):
        if self.is_loading_model:
            self._set_status("Model loading already in progress…")
            return

        if self.is_transcribing:
            messagebox.showinfo("Busy", "Please wait for current transcription to finish.")
            return

        name = self.model_var.get()
        if self.model and self.current_model_name == name:
            self._set_status(f"{name.capitalize()} already loaded")
            return

        self.is_loading_model = True

        def task():
            try:
                self._set_busy(True, f"Loading {name}…")
                self._run_on_ui(lambda: self._set_model_indicator(HEX_ORANGE, f"Loading {name}…"))

                t0 = time.time()
                self.model = whisper.load_model(name)
                self.current_model_name = name
                elapsed = time.time() - t0

                self._run_on_ui(lambda: self._set_model_indicator(HEX_GREEN, f"{name.capitalize()} ready"))
                self._set_status(f"{name.capitalize()} model loaded ({elapsed:.1f}s)")
                logger.info(f"Model {name} loaded in {elapsed:.1f}s")

            except Exception as e:
                logger.error(f"Load failed: {e}")
                self._run_on_ui(lambda: self._set_model_indicator(HEX_RED_DOT, "Load failed"))
                self._run_on_ui(lambda: messagebox.showerror(
                    "Model Error", f"Could not load model:\n\n{e}"))
                self._set_status("Error loading model")
            finally:
                self.is_loading_model = False
                self._set_busy(False)

        try:
            threading.Thread(target=task, daemon=True).start()
        except Exception as e:
            self.is_loading_model = False
            self._set_status("Failed to start model loading")
            logger.error(f"Could not start model loading thread: {e}")
            messagebox.showerror("Model Error", f"Could not start model loading:\n\n{e}")

    # ── Core: Browse / Drop ───────────────────────────────────────────────────

    def _browse_file(self):
        if self.is_transcribing or self.is_loading_model:
            messagebox.showinfo("Busy", "Please wait for transcription to complete.")
            return

        path = filedialog.askopenfilename(
            title="Open Audio or Video File",
            filetypes=[
                ("Audio/Video", " ".join(f"*{e}" for e in SUPPORTED_FORMATS)),
                ("Audio", "*.mp3 *.wav *.m4a *.ogg *.flac"),
                ("Video", "*.mp4 *.webm *.mkv *.avi *.mov"),
                ("All files", "*.*"),
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

            meta_parts = [p.suffix.upper().lstrip("."), size]
            if duration:
                meta_parts.append(format_duration(duration))

            self._show_file_info(p.name, " · ".join(meta_parts))
            self._set_status(f"Ready — {p.name}")
            logger.info(f"File: {filepath}")
        except OSError as e:
            messagebox.showerror("File Error", f"Could not read file metadata:\n\n{e}")
            logger.error(f"File metadata error for {filepath}: {e}")
            self.file_path.set("")
            self._file_duration = None

    # ── Core: Transcribe ──────────────────────────────────────────────────────

    def _transcribe(self):
        if self.is_transcribing:
            return

        if self.is_loading_model:
            messagebox.showinfo("Busy", "Please wait for model loading to finish.")
            return

        if not self.model:
            messagebox.showwarning("No Model", "Load a model first.")
            return

        filepath = self.file_path.get()
        ok, msg = validate_file(filepath)
        if not ok:
            messagebox.showwarning("No File", "Select an audio or video file first.")
            return

        language = LANGUAGES.get(self.lang_var.get())
        with_timestamps = self.timestamps_var.get()
        output_mode_display = self.output_lang_var.get()
        output_mode = OUTPUT_LANGUAGES.get(output_mode_display, "source")
        whisper_task = "translate" if output_mode == "en" else "transcribe"

        self.is_transcribing = True
        self._transcription_start = time.time()

        def task():
            try:
                self._set_busy(True, "Transcribing…")
                self._run_on_ui(self._show_transcription_ui)

                result = self.model.transcribe(
                    filepath,
                    language=language,
                    task=whisper_task,
                    verbose=False
                )
                elapsed = time.time() - self._transcription_start

                if not self.is_transcribing:
                    return

                if with_timestamps:
                    text = self._format_timestamps(result)
                else:
                    text = result["text"].strip()

                detected = result.get("language", "?")
                self._run_on_ui(lambda: self._show_result(text, elapsed, detected, output_mode_display))

            except Exception as e:
                logger.error(f"Transcription error: {e}")
                self._run_on_ui(lambda: messagebox.showerror(
                    "Transcription Failed", f"{e}\n\nCheck the file and try again."))
                self._run_on_ui(lambda: self._set_status("Transcription failed"))
            finally:
                self.is_transcribing = False
                self._run_on_ui(lambda: self._set_busy(False))
                self._run_on_ui(self._hide_transcription_ui)

        try:
            threading.Thread(target=task, daemon=True).start()
        except Exception as e:
            self.is_transcribing = False
            self._set_status("Failed to start transcription")
            logger.error(f"Could not start transcription thread: {e}")
            messagebox.showerror("Transcription Failed", f"Could not start transcription:\n\n{e}")

    def _format_timestamps(self, result):
        lines = []
        for seg in result.get("segments", []):
            s = format_duration(seg["start"])
            e = format_duration(seg["end"])
            lines.append(f"[{s} → {e}]  {seg['text'].strip()}")
        return "\n\n".join(lines)

    def _show_result(self, text, elapsed, lang, output_mode_display):
        self._clear_placeholder()
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)
        mode_label = "Translated to English" if "English" in output_mode_display else "Transcribed"
        self._set_status(
            f"{mode_label} in {format_duration(elapsed)} · Language: {lang}"
        )

    def _cancel(self):
        if self.is_transcribing:
            self.is_transcribing = False
            self._set_status("Cancelled")

    # ── Transcription UI state ────────────────────────────────────────────────

    def _show_transcription_ui(self):
        """Show cancel button, progress bar, and live elapsed timer."""
        self.cancel_btn.pack(side="left", padx=(8, 0))
        self.progress.pack(side="left", padx=(12, 0))
        self._elapsed_label.config(text="00:00")
        self._elapsed_label.pack(side="left", padx=(10, 0))
        self._action_status.config(text="Processing audio…")
        self._action_status.pack(side="left", padx=(10, 0))
        self._start_elapsed_timer()

    def _hide_transcription_ui(self):
        """Hide transcription-specific UI elements."""
        self._stop_elapsed_timer()
        self.cancel_btn.pack_forget()
        self.progress.pack_forget()
        self._elapsed_label.pack_forget()
        self._action_status.pack_forget()

    def _start_elapsed_timer(self):
        self._stop_elapsed_timer()
        def tick():
            if self.is_transcribing and self._transcription_start:
                elapsed = time.time() - self._transcription_start
                self._elapsed_label.config(text=format_elapsed_clock(elapsed))
                self._elapsed_timer_id = self.root.after(1000, tick)
        tick()

    def _stop_elapsed_timer(self):
        if self._elapsed_timer_id:
            self.root.after_cancel(self._elapsed_timer_id)
            self._elapsed_timer_id = None

    # ── Transcript actions ────────────────────────────────────────────────────

    def _save_transcript(self):
        text = self._get_transcript_text()
        if not text:
            messagebox.showwarning("Nothing to Save", "Transcript is empty.")
            return

        default = "transcript.txt"
        if self.file_path.get():
            stem = Path(self.file_path.get()).stem
            if OUTPUT_LANGUAGES.get(self.output_lang_var.get()) == "en":
                default = stem + "_english_translation.txt"
            else:
                default = stem + "_transcript.txt"

        dest = filedialog.asksaveasfilename(
            title="Save Transcript",
            defaultextension=".txt",
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
        text = self._get_transcript_text()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status("Copied to clipboard")

    def _clear_output(self):
        text = self._get_transcript_text()
        if not text:
            return
        self._show_placeholder()
        self._word_count.config(text="")
        self._set_status("Cleared")

    def _get_transcript_text(self):
        """Get transcript text, returning empty string if placeholder is active."""
        if getattr(self, '_placeholder_active', False):
            return ""
        return self.output.get("1.0", tk.END).strip()

    # ── Help & About ──────────────────────────────────────────────────────────

    def _show_help(self):
        win = tk.Toplevel(self.root)
        win.title("Whisper Transcriber Help")
        win.geometry("500x480")
        try:
            win.configure(bg=self.color_bg)
        except tk.TclError:
            win.configure(bg=HEX_BG)

        content = tk.Frame(win, bg=self.color_bg)
        content.pack(fill="both", expand=True, padx=28, pady=24)

        help_sections = [
            ("Getting Started",
             "1. Choose a model and click Load.\n"
             "   Start with base — it's fast and good enough for most audio.\n\n"
             "2. Drop or browse for an audio or video file.\n\n"
             "3. Set the language if auto-detect isn't working well.\n\n"
             "4. Choose output mode:\n"
             "   • Original language (transcribe)\n"
             "   • English (translate)\n\n"
             "5. Click Transcribe (or ⌘Return).\n\n"
             "6. Save or copy the result."),
            ("Models",
             "tiny     — quick drafts, clear recordings\n"
             "base    — good everyday balance (start here)\n"
             "small   — better accuracy, ~2x slower\n"
             "medium — high accuracy, ~6x slower\n"
             "large   — best accuracy, very slow on CPU"),
            ("Tips",
             "• Models download once on first load, then stay cached.\n"
             "• Internet is only needed for the first model download.\n"
             "• Setting the language manually can improve accuracy a lot.\n"
             "• Clear audio always gives better results.\n"
             "• Use ⌘Return to start transcribing quickly.\n"
             "• Output mode can translate speech directly to English.\n"
             "• Whisper direct translation supports English output only."),
            ("Privacy & Transparency",
             "• Transcription and translation run locally on this Mac.\n"
             "• Audio/video is NOT sent to the OpenAI API by this app.\n"
             "• No account sign-in is required for local processing."),
        ]

        for title, body in help_sections:
            tk.Label(content, text=title, font=FONT_SECTION,
                     fg=self.color_text, bg=self.color_bg, anchor="w").pack(fill="x", pady=(12, 4))
            tk.Label(content, text=body, font=("Helvetica Neue", 12),
                     fg=self.color_text, bg=self.color_bg, anchor="w", justify="left",
                     wraplength=440).pack(fill="x")

        ttk.Button(content, text="Close", command=win.destroy).pack(pady=(20, 0))

    def _show_about(self):
        messagebox.showinfo(
            "Whisper Transcriber",
            "Whisper Transcriber\n\n"
            "Local audio/video transcription\npowered by OpenAI Whisper.\n\n"
            "Privacy & Transparency:\n"
            "• Runs locally on your machine\n"
            "• Audio/video is not uploaded to OpenAI API\n"
            "• Internet is only needed for first model download"
        )

    # ── UI State Helpers ──────────────────────────────────────────────────────

    def _set_busy(self, busy, msg=None):
        def apply():
            if busy:
                self.progress.start(12)
                self.transcribe_btn.state(["disabled"])
                self.load_btn.state(["disabled"])
            else:
                self.progress.stop()
                self.transcribe_btn.state(["!disabled"])
                self.load_btn.state(["!disabled"])

        self._run_on_ui(apply)
        if msg:
            self._set_status(msg)

    def _set_status(self, msg):
        def apply():
            if hasattr(self, "_status_var"):
                self._status_var.set(msg)
        self._run_on_ui(apply)
        logger.info(f"Status: {msg}")

    def _set_model_indicator(self, color, text):
        def apply():
            if hasattr(self, "_dot") and hasattr(self, "_model_status"):
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
    logger.info("Starting Whisper Transcriber (macOS)")
    root = tk.Tk()
    WhisperTranscriber(root)
    root.mainloop()
    logger.info("Closed")

if __name__ == "__main__":
    main()
