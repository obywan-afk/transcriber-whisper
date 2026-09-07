# Icon assets

Icon source and exported files for Whisper Transcriber.

| File | Purpose |
|---|---|
| `icon.svg` | Editable vector master — edit this one |
| `icon.icns` | macOS app and DMG volume icon |
| `icon.ico` | Windows executable icon |
| `icon.iconset/` | Intermediate PNGs used to build `icon.icns` |
| `icon-512.png` | Raster copy used as the README hero image |

## Regenerating

Export the master to a 1024×1024 PNG first:

```bash
# with rsvg-convert (brew install librsvg)
rsvg-convert -w 1024 -h 1024 icon.svg -o icon-1024.png
```

### macOS `.icns`

```bash
mkdir -p icon.iconset
for s in 16 32 128 256 512; do
  sips -z $s $s      icon-1024.png --out "icon.iconset/icon_${s}x${s}.png"
  sips -z $((s*2)) $((s*2)) icon-1024.png --out "icon.iconset/icon_${s}x${s}@2x.png"
done
iconutil -c icns icon.iconset -o icon.icns
```

### Windows `.ico`

```bash
magick icon-1024.png -define icon:auto-resize=16,24,32,48,64,128,256 icon.ico
```

`magick` comes from ImageMagick — `brew install imagemagick` on macOS.
