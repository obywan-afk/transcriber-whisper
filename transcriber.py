"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        WHISPER TRANSCRIBER v1.0                              ║
║                    Audio/Video to Text Transcription                         ║
║                                                                              ║
║  Features:                                                                   ║
║  - Transcribe audio/video files to text                                      ║
║  - Multiple language support (auto-detect or manual)                         ║
║  - Multiple model sizes (speed vs accuracy trade-off)                        ║
║  - Save transcripts to text files                                            ║
║                                                                              ║
║  Supported formats: MP3, WAV, M4A, OGG, FLAC, MP4, WEBM, MKV, AVI           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FIX FOR PYINSTALLER WINDOWED MODE - MUST BE AT THE VERY TOP
# ═══════════════════════════════════════════════════════════════════════════════

import sys
import os

# Redirect stdout/stderr to devnull if they don't exist (windowed mode)
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
import threading
import subprocess
import logging
import time
from datetime import datetime
from pathlib import Path

# Suppress FP16 warning
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

log_dir = Path.home() / ".whisper_transcriber"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"transcriber_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file, encoding='utf-8')]
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# FFMPEG SETUP
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import numpy as np
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    logger.info(f"FFmpeg found at: {ffmpeg_exe}")
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    ffmpeg_exe = None

# ═══════════════════════════════════════════════════════════════════════════════
# WHISPER SETUP WITH PATCHED AUDIO LOADING
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import whisper
    import whisper.audio
    
    def _patched_load_audio(file, sr=16000):
        """Load audio file using FFmpeg with explicit path."""
        if not ffmpeg_exe:
            raise RuntimeError("FFmpeg not found. Please reinstall imageio-ffmpeg.")
        
        if not os.path.exists(file):
            raise FileNotFoundError(f"Audio file not found: {file}")
        
        cmd = [
            ffmpeg_exe,
            "-nostdin",
            "-threads", "0",
            "-i", file,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-"
        ]
        
        try:
            # Fix for PyInstaller --windowed mode (no console)
            startupinfo = None
            if os.name == 'nt':  # Windows
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                startupinfo=startupinfo
            )
            out, err = process.communicate()
            
            if process.returncode != 0:
                error_msg = err.decode() if err else "Unknown error"
                raise RuntimeError(f"FFmpeg failed to process audio: {error_msg}")
                
        except Exception as e:
            raise RuntimeError(f"FFmpeg failed to process audio: {str(e)}")
        
        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    
    whisper.audio.load_audio = _patched_load_audio
    logger.info("Whisper loaded successfully")
    
except ImportError as e:
    logger.error(f"Failed to import Whisper: {e}")
    whisper = None

# ABB Brand Colors
ABB_RED = "#FF000F"
ABB_RED_HOVER = "#CC000C"
ABB_DARK = "#1F1F1F"
ABB_GRAY = "#6E6E6E"
ABB_LIGHT_GRAY = "#F5F5F5"

# Model information
MODEL_INFO = {
    "tiny": {
        "size": "~39 MB",
        "speed": "Fastest",
        "accuracy": "Basic",
        "description": "Best for quick drafts or clear audio",
        "vram": "~1 GB"
    },
    "base": {
        "size": "~74 MB", 
        "speed": "Fast",
        "accuracy": "Good",
        "description": "Good balance for everyday use",
        "vram": "~1 GB"
    },
    "small": {
        "size": "~244 MB",
        "speed": "Medium",
        "accuracy": "Better",
        "description": "Recommended for important transcripts",
        "vram": "~2 GB"
    },
    "medium": {
        "size": "~769 MB",
        "speed": "Slow",
        "accuracy": "Great",
        "description": "High accuracy, requires patience",
        "vram": "~5 GB"
    },
    "large": {
        "size": "~1550 MB",
        "speed": "Very Slow",
        "accuracy": "Best",
        "description": "Maximum accuracy, very slow on CPU",
        "vram": "~10 GB"
    }
}

# Supported file formats
SUPPORTED_FORMATS = (
    ".mp3", ".wav", ".m4a", ".ogg", ".flac",  # Audio
    ".mp4", ".webm", ".mkv", ".avi", ".mov"    # Video
)

# Common languages
LANGUAGES = {
    "Auto-detect": "auto",
    "English": "en",
    "Finnish": "fi",
    "Swedish": "sv",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Dutch": "nl",
    "Polish": "pl",
    "Portuguese": "pt",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko"
}

# ═══════════════════════════════════════════════════════════════════════════════
# TOOLTIP HELPER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class ToolTip:
    """Create a tooltip for a given widget."""
    
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_window = None
        self.scheduled_id = None
        
        widget.bind("<Enter>", self._schedule_tooltip)
        widget.bind("<Leave>", self._hide_tooltip)
        widget.bind("<ButtonPress>", self._hide_tooltip)
    
    def _schedule_tooltip(self, event=None):
        self._hide_tooltip()
        self.scheduled_id = self.widget.after(self.delay, self._show_tooltip)
    
    def _show_tooltip(self):
        if self.tooltip_window:
            return
        
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        # Style the tooltip
        frame = tk.Frame(tw, background="#333333", borderwidth=1, relief="solid")
        frame.pack()
        
        label = tk.Label(
            frame,
            text=self.text,
            justify="left",
            background="#333333",
            foreground="white",
            font=("Segoe UI", 9),
            padx=8,
            pady=5,
            wraplength=300
        )
        label.pack()
    
    def _hide_tooltip(self, event=None):
        if self.scheduled_id:
            self.widget.after_cancel(self.scheduled_id)
            self.scheduled_id = None
        
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_file(filepath):
    """Validate that file exists and is a supported format."""
    if not filepath:
        return False, "No file selected"
    
    path = Path(filepath)
    
    if not path.exists():
        return False, f"File not found: {filepath}"
    
    if not path.is_file():
        return False, f"Not a file: {filepath}"
    
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        return False, f"Unsupported format: {path.suffix}\n\nSupported: {', '.join(SUPPORTED_FORMATS)}"
    
    return True, "OK"

def format_duration(seconds):
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"

def get_file_duration(filepath):
    """Get audio/video file duration using FFmpeg."""
    if not ffmpeg_exe:
        return None
    
    try:
        cmd = [
            ffmpeg_exe,
            "-i", filepath,
            "-hide_banner"
        ]
        
        # Fix for PyInstaller --windowed mode
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo,
            text=True
        )
        
        # Parse duration from stderr (FFmpeg outputs info to stderr)
        for line in result.stderr.split('\n'):
            if 'Duration:' in line:
                time_str = line.split('Duration:')[1].split(',')[0].strip()
                parts = time_str.split(':')
                hours, mins, secs = float(parts[0]), float(parts[1]), float(parts[2])
                return hours * 3600 + mins * 60 + secs
    except Exception as e:
        logger.warning(f"Could not get file duration: {e}")
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class WhisperTranscriber:
    """Main application class for Whisper Transcriber."""
    
    def __init__(self, root):
        self.root = root
        self.model = None
        self.current_model_name = None
        self.is_transcribing = False
        self.transcription_start_time = None
        
        # Configure window
        self._setup_window()
        
        # Check dependencies
        if not self._check_dependencies():
            return
        
        # Build UI
        self._create_header()
        self._create_model_section()
        self._create_file_section()
        self._create_options_section()
        self._create_action_section()
        self._create_output_section()
        self._create_footer()
        
        logger.info("Application initialized successfully")
    
    def _setup_window(self):
        """Configure the main window."""
        self.root.title("Whisper Transcriber")
        self.root.geometry("850x700")
        self.root.minsize(700, 600)
        self.root.configure(bg="white")
        
        # Center window on screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - 850) // 2
        y = (self.root.winfo_screenheight() - 700) // 2
        self.root.geometry(f"+{x}+{y}")
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure("TCombobox", padding=5)
        self.style.configure("TProgressbar", thickness=8)
    
    def _check_dependencies(self):
        """Check if all required dependencies are available."""
        errors = []
        
        if not ffmpeg_exe:
            errors.append("FFmpeg not found. Please install imageio-ffmpeg.")
        
        if whisper is None:
            errors.append("Whisper not found. Please install openai-whisper.")
        
        if errors:
            error_msg = "Missing dependencies:\n\n" + "\n".join(f"• {e}" for e in errors)
            error_msg += "\n\nPlease run:\nuv pip install openai-whisper imageio-ffmpeg"
            
            messagebox.showerror("Dependency Error", error_msg)
            logger.error(f"Dependency check failed: {errors}")
            return False
        
        return True
    
    # ─────────────────────────────────────────────────────────────────────────
    # UI CREATION METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    def _create_header(self):
        """Create the application header."""
        header = tk.Frame(self.root, bg=ABB_RED, height=60)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        title = tk.Label(
            header,
            text="🎙️  Whisper Transcriber",
            font=("Segoe UI", 18, "bold"),
            bg=ABB_RED,
            fg="white"
        )
        title.pack(side="left", padx=20, pady=15)
        
        # Help button
        help_btn = tk.Button(
            header,
            text="❓ Help",
            font=("Segoe UI", 10),
            bg="white",
            fg=ABB_RED,
            relief="flat",
            cursor="hand2",
            command=self._show_help
        )
        help_btn.pack(side="right", padx=20, pady=15)
    
    def _create_model_section(self):
        """Create the model selection section."""
        section = tk.LabelFrame(
            self.root,
            text=" Step 1: Select & Load Model ",
            font=("Segoe UI", 11, "bold"),
            fg=ABB_DARK,
            bg="white",
            padx=15,
            pady=10
        )
        section.pack(fill="x", padx=20, pady=(15, 10))
        
        # Model selection row
        row1 = tk.Frame(section, bg="white")
        row1.pack(fill="x", pady=5)
        
        tk.Label(
            row1,
            text="Model:",
            font=("Segoe UI", 10),
            bg="white",
            width=10,
            anchor="w"
        ).pack(side="left")
        
        self.model_var = tk.StringVar(value="base")
        self.model_dropdown = ttk.Combobox(
            row1,
            textvariable=self.model_var,
            values=list(MODEL_INFO.keys()),
            width=12,
            state="readonly"
        )
        self.model_dropdown.pack(side="left", padx=(0, 10))
        self.model_dropdown.bind("<<ComboboxSelected>>", self._on_model_selected)
        
        ToolTip(self.model_dropdown, "Choose a model based on your needs:\n• Smaller = Faster but less accurate\n• Larger = Slower but more accurate")
        
        self.load_btn = tk.Button(
            row1,
            text="📥 Load Model",
            font=("Segoe UI", 10, "bold"),
            bg=ABB_RED,
            fg="white",
            relief="flat",
            cursor="hand2",
            padx=15,
            pady=5,
            command=self._load_model
        )
        self.load_btn.pack(side="left", padx=(0, 15))
        
        ToolTip(self.load_btn, "Load the selected model into memory.\nFirst load downloads the model (one-time).")
        
        # Status indicator
        self.model_status_frame = tk.Frame(row1, bg="white")
        self.model_status_frame.pack(side="left", fill="x", expand=True)
        
        self.model_status_indicator = tk.Label(
            self.model_status_frame,
            text="●",
            font=("Segoe UI", 14),
            fg="#CC0000",
            bg="white"
        )
        self.model_status_indicator.pack(side="left")
        
        self.model_status_text = tk.Label(
            self.model_status_frame,
            text="No model loaded",
            font=("Segoe UI", 10),
            fg=ABB_GRAY,
            bg="white"
        )
        self.model_status_text.pack(side="left", padx=5)
        
        # Model info row
        self.model_info_label = tk.Label(
            section,
            text="ℹ️  Base model: ~74 MB | Fast | Good accuracy | Good balance for everyday use",
            font=("Segoe UI", 9),
            fg=ABB_GRAY,
            bg="white",
            anchor="w"
        )
        self.model_info_label.pack(fill="x", pady=(5, 0))
    
    def _create_file_section(self):
        """Create the file selection section."""
        section = tk.LabelFrame(
            self.root,
            text=" Step 2: Select Audio/Video File ",
            font=("Segoe UI", 11, "bold"),
            fg=ABB_DARK,
            bg="white",
            padx=15,
            pady=10
        )
        section.pack(fill="x", padx=20, pady=10)
        
        row = tk.Frame(section, bg="white")
        row.pack(fill="x", pady=5)
        
        tk.Label(
            row,
            text="File:",
            font=("Segoe UI", 10),
            bg="white",
            width=10,
            anchor="w"
        ).pack(side="left")
        
        self.file_path = tk.StringVar()
        self.file_entry = tk.Entry(
            row,
            textvariable=self.file_path,
            font=("Segoe UI", 10),
            width=50,
            state="readonly"
        )
        self.file_entry.pack(side="left", padx=(0, 10), fill="x", expand=True)
        
        self.browse_btn = tk.Button(
            row,
            text="📂 Browse...",
            font=("Segoe UI", 10),
            bg=ABB_LIGHT_GRAY,
            fg=ABB_DARK,
            relief="flat",
            cursor="hand2",
            padx=15,
            pady=5,
            command=self._browse_file
        )
        self.browse_btn.pack(side="left")
        
        ToolTip(self.browse_btn, f"Supported formats:\n{', '.join(SUPPORTED_FORMATS)}")
        
        # File info row
        self.file_info_label = tk.Label(
            section,
            text="",
            font=("Segoe UI", 9),
            fg=ABB_GRAY,
            bg="white",
            anchor="w"
        )
        self.file_info_label.pack(fill="x", pady=(5, 0))
    
    def _create_options_section(self):
        """Create the options section."""
        section = tk.LabelFrame(
            self.root,
            text=" Step 3: Options (Optional) ",
            font=("Segoe UI", 11, "bold"),
            fg=ABB_DARK,
            bg="white",
            padx=15,
            pady=10
        )
        section.pack(fill="x", padx=20, pady=10)
        
        row = tk.Frame(section, bg="white")
        row.pack(fill="x", pady=5)
        
        # Language selection
        tk.Label(
            row,
            text="Language:",
            font=("Segoe UI", 10),
            bg="white",
            width=10,
            anchor="w"
        ).pack(side="left")
        
        self.lang_var = tk.StringVar(value="Auto-detect")
        self.lang_dropdown = ttk.Combobox(
            row,
            textvariable=self.lang_var,
            values=list(LANGUAGES.keys()),
            width=15,
            state="readonly"
        )
        self.lang_dropdown.pack(side="left", padx=(0, 30))
        
        ToolTip(self.lang_dropdown, "Select the spoken language in the audio.\n'Auto-detect' works well for most cases.\nManual selection can improve accuracy.")
        
        # Timestamps checkbox
        self.timestamps_var = tk.BooleanVar(value=False)
        self.timestamps_check = tk.Checkbutton(
            row,
            text="Include timestamps",
            variable=self.timestamps_var,
            font=("Segoe UI", 10),
            bg="white",
            cursor="hand2"
        )
        self.timestamps_check.pack(side="left")
        
        ToolTip(self.timestamps_check, "Add time markers to the transcript.\nUseful for referencing specific parts.")
    
    def _create_action_section(self):
        """Create the main action buttons section."""
        section = tk.Frame(self.root, bg="white", pady=10)
        section.pack(fill="x", padx=20)
        
        # Transcribe button (large, prominent)
        self.transcribe_btn = tk.Button(
            section,
            text="▶️  TRANSCRIBE",
            font=("Segoe UI", 14, "bold"),
            bg=ABB_RED,
            fg="white",
            relief="flat",
            cursor="hand2",
            padx=40,
            pady=12,
            command=self._transcribe
        )
        self.transcribe_btn.pack(side="left", padx=(0, 15))
        
        ToolTip(self.transcribe_btn, "Start transcription.\nMake sure a model is loaded and a file is selected.")
        
        # Cancel button (initially hidden)
        self.cancel_btn = tk.Button(
            section,
            text="⏹️ Cancel",
            font=("Segoe UI", 11),
            bg=ABB_GRAY,
            fg="white",
            relief="flat",
            cursor="hand2",
            padx=20,
            pady=12,
            command=self._cancel_transcription
        )
        # Don't pack yet - shown only during transcription
        
        # Progress section
        progress_frame = tk.Frame(section, bg="white")
        progress_frame.pack(side="left", fill="x", expand=True, padx=10)
        
        self.progress = ttk.Progressbar(
            progress_frame,
            mode="indeterminate",
            length=200
        )
        self.progress.pack(fill="x", pady=(0, 5))
        
        self.status_var = tk.StringVar(value="Ready to transcribe")
        self.status_label = tk.Label(
            progress_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            fg=ABB_GRAY,
            bg="white",
            anchor="w"
        )
        self.status_label.pack(fill="x")
    
    def _create_output_section(self):
        """Create the output/transcript section."""
        section = tk.LabelFrame(
            self.root,
            text=" Transcript ",
            font=("Segoe UI", 11, "bold"),
            fg=ABB_DARK,
            bg="white",
            padx=15,
            pady=10
        )
        section.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Toolbar
        toolbar = tk.Frame(section, bg="white")
        toolbar.pack(fill="x", pady=(0, 10))
        
        self.save_btn = tk.Button(
            toolbar,
            text="💾 Save to File",
            font=("Segoe UI", 10),
            bg=ABB_LIGHT_GRAY,
            fg=ABB_DARK,
            relief="flat",
            cursor="hand2",
            padx=10,
            pady=3,
            command=self._save_transcript
        )
        self.save_btn.pack(side="left", padx=(0, 10))
        
        ToolTip(self.save_btn, "Save the transcript as a text file.")
        
        self.copy_btn = tk.Button(
            toolbar,
            text="📋 Copy All",
            font=("Segoe UI", 10),
            bg=ABB_LIGHT_GRAY,
            fg=ABB_DARK,
            relief="flat",
            cursor="hand2",
            padx=10,
            pady=3,
            command=self._copy_transcript
        )
        self.copy_btn.pack(side="left", padx=(0, 10))
        
        ToolTip(self.copy_btn, "Copy the entire transcript to clipboard.")
        
        self.clear_btn = tk.Button(
            toolbar,
            text="🗑️ Clear",
            font=("Segoe UI", 10),
            bg=ABB_LIGHT_GRAY,
            fg=ABB_DARK,
            relief="flat",
            cursor="hand2",
            padx=10,
            pady=3,
            command=self._clear_output
        )
        self.clear_btn.pack(side="left")
        
        ToolTip(self.clear_btn, "Clear the transcript output.")
        
        # Word count label
        self.word_count_var = tk.StringVar(value="")
        self.word_count_label = tk.Label(
            toolbar,
            textvariable=self.word_count_var,
            font=("Segoe UI", 9),
            fg=ABB_GRAY,
            bg="white"
        )
        self.word_count_label.pack(side="right")
        
        # Text output
        self.output = scrolledtext.ScrolledText(
            section,
            font=("Consolas", 11),
            wrap=tk.WORD,
            bg="#FAFAFA",
            relief="solid",
            borderwidth=1
        )
        self.output.pack(fill="both", expand=True)
        self.output.bind("<<Modified>>", self._on_text_modified)
    
    def _create_footer(self):
        """Create the application footer."""
        footer = tk.Frame(self.root, bg=ABB_LIGHT_GRAY, height=30)
        footer.pack(fill="x", side="bottom")
        
        tk.Label(
            footer,
            text="Powered by OpenAI Whisper",
            font=("Segoe UI", 8),
            fg=ABB_GRAY,
            bg=ABB_LIGHT_GRAY
        ).pack(side="left", padx=10, pady=5)
        
        tk.Label(
            footer,
            text="v1.0",
            font=("Segoe UI", 8),
            fg=ABB_GRAY,
            bg=ABB_LIGHT_GRAY
        ).pack(side="right", padx=10, pady=5)
    
    # ─────────────────────────────────────────────────────────────────────────
    # EVENT HANDLERS
    # ─────────────────────────────────────────────────────────────────────────
    
    def _on_model_selected(self, event=None):
        """Handle model selection change."""
        model_name = self.model_var.get()
        info = MODEL_INFO.get(model_name, {})
        
        info_text = f"ℹ️  {model_name.capitalize()} model: {info.get('size', '?')} | {info.get('speed', '?')} | {info.get('accuracy', '?')} accuracy | {info.get('description', '')}"
        self.model_info_label.config(text=info_text)
        
        logger.info(f"Model selected: {model_name}")
    
    def _on_text_modified(self, event=None):
        """Handle text modification in output."""
        if self.output.edit_modified():
            text = self.output.get(1.0, tk.END).strip()
            words = len(text.split()) if text else 0
            chars = len(text)
            self.word_count_var.set(f"{words} words | {chars} characters")
            self.output.edit_modified(False)
    
    # ─────────────────────────────────────────────────────────────────────────
    # CORE FUNCTIONALITY
    # ─────────────────────────────────────────────────────────────────────────
    
    def _load_model(self):
        """Load the selected Whisper model."""
        model_name = self.model_var.get()
        
        # Check if same model is already loaded
        if self.model and self.current_model_name == model_name:
            messagebox.showinfo("Already Loaded", f"The {model_name} model is already loaded and ready.")
            return
        
        def load_task():
            try:
                self._set_loading_state(True, f"Loading {model_name} model...")
                self.model_status_indicator.config(fg="#FFA500")  # Orange - loading
                self.model_status_text.config(text=f"Loading {model_name}...")
                
                logger.info(f"Loading model: {model_name}")
                start_time = time.time()
                
                self.model = whisper.load_model(model_name)
                self.current_model_name = model_name
                
                elapsed = time.time() - start_time
                logger.info(f"Model loaded in {elapsed:.1f}s")
                
                # Update UI
                self.model_status_indicator.config(fg="#00AA00")  # Green - loaded
                self.model_status_text.config(text=f"{model_name.capitalize()} model ready ✓")
                self._set_status(f"Model loaded successfully ({elapsed:.1f}s)")
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model_status_indicator.config(fg="#CC0000")  # Red - error
                self.model_status_text.config(text="Failed to load")
                messagebox.showerror("Error", f"Failed to load model:\n\n{str(e)}")
                self._set_status("Error loading model")
            finally:
                self._set_loading_state(False)
        
        threading.Thread(target=load_task, daemon=True).start()
    
    def _browse_file(self):
        """Open file browser to select audio/video file."""
        filetypes = [
            ("Audio/Video files", " ".join(f"*{ext}" for ext in SUPPORTED_FORMATS)),
            ("Audio files", "*.mp3 *.wav *.m4a *.ogg *.flac"),
            ("Video files", "*.mp4 *.webm *.mkv *.avi *.mov"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select Audio or Video File",
            filetypes=filetypes
        )
        
        if filepath:
            self._set_file(filepath)
    
    def _set_file(self, filepath):
        """Set the selected file and update UI."""
        is_valid, message = validate_file(filepath)
        
        if not is_valid:
            messagebox.showerror("Invalid File", message)
            logger.warning(f"Invalid file selected: {message}")
            return
        
        self.file_path.set(filepath)
        
        # Get file info
        path = Path(filepath)
        size_mb = path.stat().st_size / (1024 * 1024)
        duration = get_file_duration(filepath)
        
        info_parts = [f"📄 {path.name}", f"Size: {size_mb:.1f} MB"]
        if duration:
            info_parts.append(f"Duration: {format_duration(duration)}")
        
        self.file_info_label.config(text=" | ".join(info_parts))
        logger.info(f"File selected: {filepath}")
    
    def _transcribe(self):
        """Start the transcription process."""
        # Validation
        if not self.model:
            messagebox.showwarning(
                "No Model Loaded",
                "Please load a model first (Step 1).\n\nClick 'Load Model' to continue."
            )
            return
        
        filepath = self.file_path.get()
        is_valid, message = validate_file(filepath)
        
        if not is_valid:
            messagebox.showwarning(
                "No File Selected", 
                "Please select an audio or video file (Step 2).\n\nClick 'Browse...' to select a file."
            )
            return
        
        # Get options
        lang_display = self.lang_var.get()
        language = LANGUAGES.get(lang_display)
        if language == "auto":
            language = None
        
        include_timestamps = self.timestamps_var.get()
        
        def transcribe_task():
            try:
                self.is_transcribing = True
                self._set_loading_state(True, "Transcribing... This may take a while.")
                self.transcription_start_time = time.time()
                
                # Show cancel button
                self.root.after(0, lambda: self.cancel_btn.pack(side="left", padx=(0, 15)))
                
                logger.info(f"Starting transcription: {filepath}, language={language}, timestamps={include_timestamps}")
                
                # Perform transcription
                result = self.model.transcribe(
                    filepath,
                    language=language,
                    verbose=False
                )
                
                elapsed = time.time() - self.transcription_start_time
                
                if not self.is_transcribing:  # Was cancelled
                    return
                
                # Format output
                if include_timestamps:
                    output_text = self._format_with_timestamps(result)
                else:
                    output_text = result["text"].strip()
                
                # Update UI
                self.root.after(0, lambda: self._display_result(output_text, elapsed, result))
                
                logger.info(f"Transcription completed in {elapsed:.1f}s")
                
            except Exception as e:
                logger.error(f"Transcription failed: {e}")
                self.root.after(0, lambda: messagebox.showerror(
                    "Transcription Failed",
                    f"An error occurred during transcription:\n\n{str(e)}\n\nPlease check the file and try again."
                ))
                self.root.after(0, lambda: self._set_status("Transcription failed"))
            finally:
                self.is_transcribing = False
                self.root.after(0, lambda: self._set_loading_state(False))
                self.root.after(0, lambda: self.cancel_btn.pack_forget())
        
        threading.Thread(target=transcribe_task, daemon=True).start()
    
    def _format_with_timestamps(self, result):
        """Format transcription result with timestamps."""
        lines = []
        for segment in result.get("segments", []):
            start = format_duration(segment["start"])
            end = format_duration(segment["end"])
            text = segment["text"].strip()
            lines.append(f"[{start} → {end}]  {text}")
        return "\n\n".join(lines)
    
    def _display_result(self, text, elapsed, result):
        """Display the transcription result."""
        self.output.delete(1.0, tk.END)
        self.output.insert(tk.END, text)
        
        detected_lang = result.get("language", "unknown")
        self._set_status(f"✓ Completed in {format_duration(elapsed)} | Detected language: {detected_lang}")
    
    def _cancel_transcription(self):
        """Cancel the current transcription."""
        if self.is_transcribing:
            self.is_transcribing = False
            self._set_status("Transcription cancelled")
            logger.info("Transcription cancelled by user")
    
    def _save_transcript(self):
        """Save the transcript to a file."""
        text = self.output.get(1.0, tk.END).strip()
        
        if not text:
            messagebox.showwarning("Nothing to Save", "The transcript is empty.\n\nTranscribe a file first.")
            return
        
        # Suggest filename based on input file
        default_name = "transcript.txt"
        if self.file_path.get():
            input_name = Path(self.file_path.get()).stem
            default_name = f"{input_name}_transcript.txt"
        
        filepath = filedialog.asksaveasfilename(
            title="Save Transcript",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(text)
                self._set_status(f"Saved to: {Path(filepath).name}")
                logger.info(f"Transcript saved to: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save transcript: {e}")
                messagebox.showerror("Save Failed", f"Could not save file:\n\n{str(e)}")
    
    def _copy_transcript(self):
        """Copy transcript to clipboard."""
        text = self.output.get(1.0, tk.END).strip()
        
        if not text:
            messagebox.showinfo("Nothing to Copy", "The transcript is empty.")
            return
        
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status("Copied to clipboard ✓")
    
    def _clear_output(self):
        """Clear the transcript output."""
        if self.output.get(1.0, tk.END).strip():
            if messagebox.askyesno("Clear Transcript", "Are you sure you want to clear the transcript?"):
                self.output.delete(1.0, tk.END)
                self.word_count_var.set("")
                self._set_status("Output cleared")
    
    def _show_help(self):
        """Show help dialog."""
        help_text = """
WHISPER TRANSCRIBER - QUICK GUIDE
═══════════════════════════════════════

HOW TO USE:
1️⃣  Select a model and click "Load Model"
    • Start with "base" for a good balance
    • Use "tiny" for quick drafts
    • Use "small" or "medium" for important work

2️⃣  Click "Browse" to select your audio/video file
    • Supported: MP3, WAV, M4A, MP4, and more

3️⃣  (Optional) Set language if auto-detect fails

4️⃣  Click "TRANSCRIBE" and wait

5️⃣  Save or copy your transcript


TIPS:
• First model load downloads files (one-time)
• Longer files take more time
• Clear audio = better results
• Manually selecting language can improve accuracy


TROUBLESHOOTING:
• "File not found" → Check file path has no special characters
• Slow transcription → Use a smaller model (tiny/base)
• Poor accuracy → Try a larger model or set language manually
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Help")
        help_window.geometry("500x550")
        help_window.configure(bg="white")
        
        text_widget = scrolledtext.ScrolledText(
            help_window,
            font=("Consolas", 10),
            wrap=tk.WORD,
            bg="white",
            relief="flat"
        )
        text_widget.pack(fill="both", expand=True, padx=20, pady=20)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state="disabled")
        
        close_btn = tk.Button(
            help_window,
            text="Close",
            command=help_window.destroy,
            bg=ABB_RED,
            fg="white",
            font=("Segoe UI", 10),
            relief="flat",
            padx=20,
            pady=5
        )
        close_btn.pack(pady=(0, 20))
    
    # ─────────────────────────────────────────────────────────────────────────
    # UI HELPER METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    def _set_loading_state(self, is_loading, message=None):
        """Set UI to loading/ready state."""
        if is_loading:
            self.progress.start(10)
            self.transcribe_btn.config(state="disabled", bg=ABB_GRAY)
            self.load_btn.config(state="disabled")
            self.browse_btn.config(state="disabled")
            if message:
                self._set_status(message)
        else:
            self.progress.stop()
            self.transcribe_btn.config(state="normal", bg=ABB_RED)
            self.load_btn.config(state="normal")
            self.browse_btn.config(state="normal")
    
    def _set_status(self, message):
        """Update status message."""
        self.status_var.set(message)
        logger.info(f"Status: {message}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Starting Whisper Transcriber")
    logger.info("=" * 60)
    
    root = tk.Tk()
    app = WhisperTranscriber(root)
    root.mainloop()
    
    logger.info("Application closed")

if __name__ == "__main__":
    main()
