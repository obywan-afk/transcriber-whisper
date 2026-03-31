# Whisper Transcriber (macOS + Windows)

Local-first audio/video **transcriber + translator** powered by OpenAI Whisper.

- ✅ Works on **macOS** and **Windows**
- ✅ Supports audio + video files
- ✅ Transcription in original language or direct translation to English
- ✅ Runs locally (no cloud upload of your media)
- ✅ Can be distributed as a **no-install executable**

---

## Project files

- `transcriber_mac.py` → macOS UI version
- `transcriber_windows.py` → Windows UI version
- `assets/icon.svg` → source icon (editable)
- `assets/icon.icns` → macOS app icon
- `assets/icon.ico` → Windows executable icon

---

## Features

- Drag/drop or browse for media files
- Supported formats: `mp3, wav, m4a, ogg, flac, mp4, webm, mkv, avi, mov`
- Whisper model picker (tiny → large)
- Optional timestamps
- Save transcript to `.txt`
- Local processing with FFmpeg + Whisper

---

## Run from source (dev mode)

### 1) Install Python dependencies

```bash
pip install openai-whisper imageio-ffmpeg numpy
```

> On first model load, Whisper downloads model files once and caches them locally.

### 2) Start app

**macOS:**
```bash
python transcriber_mac.py
```

**Windows:**
```bash
python transcriber_windows.py
```

---

## Build no-install executables (recommended)

Use PyInstaller to generate standalone apps users can run without installing Python.

### Install build dependency

```bash
pip install pyinstaller
```

### macOS `.app`

```bash
pyinstaller --noconfirm --windowed --name WhisperTranscriber \
  --icon assets/icon.icns \
  transcriber_mac.py
```

Output:
- `dist/WhisperTranscriber.app`

### Windows `.exe`

```bash
pyinstaller --noconfirm --windowed --name WhisperTranscriber \
  --icon assets/icon.ico \
  transcriber_windows.py
```

Output:
- `dist/WhisperTranscriber/WhisperTranscriber.exe` (onedir)

> If you want one-file output, add `--onefile` (slower startup, temp extraction).

---

## App icon

Icon assets are provided in multiple formats:

- `assets/icon.svg` → source vector icon (editable)
- `assets/icon.icns` → macOS app icon
- `assets/icon.ico` → Windows executable icon

To regenerate icons from SVG, see `assets/README.md`.

---

## Git + distribution strategy (best practice)

Yes — you **should upload this to GitHub**.

Recommended setup:

1. **GitHub repo** for source code + issue tracking
2. **GitHub Releases** for compiled binaries (`.app` / `.exe` zip files)
3. Optional **GitHub Actions** to auto-build release artifacts

This gives easy sharing + version history + rollback + cleaner updates.

---

## First-time Git upload commands

From the project root:

```bash
git init
git add .
git commit -m "Initial release: macOS + Windows Whisper Transcriber"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

Then create a release tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```

Upload zipped executables to that release.

---

## What to upload for users (no installation flow)

### macOS

- Zip `WhisperTranscriber.app`
- Users download + unzip + open app
- If unsigned, users may need Gatekeeper override once

### Windows

- Zip the `dist/WhisperTranscriber` folder (or single `.exe` if using `--onefile`)
- Users unzip and run
- SmartScreen warning may appear if app is not code-signed

---

## Privacy note

This app performs transcription locally using Whisper.

- Media is processed on-device
- Media is **not uploaded to OpenAI API** by this app
- Internet is only needed when downloading models the first time
