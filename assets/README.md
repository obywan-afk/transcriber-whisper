# Icon assets

This folder contains icon source files for Whisper Transcriber.

## Source

- `icon.svg` — editable vector master icon

## Recommended export pipeline

1. Export `icon.svg` to `icon-1024.png`
2. Generate macOS `.icns`
3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Wi3. Generate Winset/icon_512x512.png
cp icon-1024.png  icon.iconset/icon_512x512@2x.png
iconutil -c icns icon.iconset
mv icon.icns ../icon.icns
```

---

## Windows `.ico` generation (ImageMagick)

```bash
magick icon-1024.png -define icon:auto-resize=16,24,32,48,64,128,256 icon.ico
```

If `magick` is missing on macOS:

```bash
brew install imagemagick
```
