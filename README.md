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

## Download

Latest macOS build (Apple Silicon):

**[WhisperTranscriber-mac.dmg](https://github.com/obywan-afk/transcriber-whisper/releases/download/latest-mac/WhisperTranscriber-mac.dmg)**

Open the DMG and drag the app to Applications.

If macOS says the app "cannot be opened", the build was not notarized. Right-click
the app in Applications, choose **Open**, and confirm. You only need to do this
once. The equivalent from a terminal is:

```bash
xattr -dr com.apple.quarantine /Applications/WhisperTranscriber.app
```

Every build is also attached to the [releases page](https://github.com/obywan-afk/transcriber-whisper/releases),
with a SHA-256 checksum in the release notes.

There is no Windows build yet — see [Run from source](#run-from-source) below.

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

The builds are written to `dist/`.

The macOS app is built automatically by
[`.github/workflows/build-mac.yml`](.github/workflows/build-mac.yml) on every push
to `main`, which signs it, notarizes it when the Apple Developer credentials allow,
and publishes the DMG to the `latest-mac` release. Pushing a `v*` tag publishes a
numbered release with the same DMG attached.

## Files

- `transcriber_mac.py` — macOS interface
- `transcriber_windows.py` — Windows interface
- `assets/` — editable source icon plus macOS and Windows icon files

See `assets/README.md` for regenerating the icons.
