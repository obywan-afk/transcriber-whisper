# Whisper Transcriber

A small local desktop app for transcribing and translating audio or video with
[Whisper](https://github.com/openai/whisper). It has separate macOS and Windows
interfaces and keeps media on the device.

## What it does

- Opens audio and video files, including `mp3`, `wav`, `m4a`, `mp4`, `mov` and `mkv`
- Transcribes in the original language or translates directly to English
- Lets you choose a Whisper model and optionally include timestamps
- Saves the result as a `.txt` file

Models download on first use and are then cached locally. The app does not send
media to the OpenAI API or another cloud service.

## Run from source

Install the dependencies:

```bash
pip install openai-whisper imageio-ffmpeg numpy
```

Then run the version for your operating system:

```bash
# macOS
python transcriber_mac.py

# Windows
python transcriber_windows.py
```

## Build an app

The repository includes icon assets and PyInstaller commands for packaging a
no-install app. Build on the operating system you are targeting:

```bash
pip install pyinstaller

# macOS
pyinstaller --noconfirm --windowed --name WhisperTranscriber \
  --icon assets/icon.icns transcriber_mac.py

# Windows
pyinstaller --noconfirm --windowed --name WhisperTranscriber \
  --icon assets/icon.ico transcriber_windows.py
```

The builds are written to `dist/`. There are no maintained downloadable binaries
yet, so please build from source for now.

## Files

- `transcriber_mac.py` — macOS interface
- `transcriber_windows.py` — Windows interface
- `assets/` — editable source icon plus macOS and Windows icon files

See `assets/README.md` for regenerating the icons.
