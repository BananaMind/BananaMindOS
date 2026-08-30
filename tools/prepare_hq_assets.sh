#!/bin/sh
set -eu

output=${1:-build/hq-assets}
source_dir=${2:-assets/hq/source}

mkdir -p "$output"

# The catalog has a fixed 1024x768 design surface. GRUB requests that mode for
# the high-quality entry, while the kernel still clips safely on a fallback mode.
magick "$source_dir/background.jpg" \
    -auto-orient -resize '1024x768^' -gravity center -extent 1024x768 \
    -colorspace sRGB -quality 90 "$output/BACK.QOI"

# Remove only the logo's border-connected white field; internal white details
# remain intact. ImageMagick's fuzzy flood fill is deterministic for this asset.
magick "$source_dir/banana.webp" -alpha on -bordercolor white -border 1 \
    -fuzz 10% -fill none -draw 'alpha 0,0 floodfill' -shave 1x1 \
    -trim +repage -resize '58x58>' -gravity center -background none \
    -extent 58x58 "$output/BANANA.QOI"

magick -background none "$source_dir/cpu.svg" -trim +repage \
    -resize '36x36>' -gravity center -extent 40x40 "$output/CPU.QOI"
magick "$source_dir/cube.png" -trim +repage -resize '34x34>' \
    -gravity center -background none -extent 40x40 "$output/CUBE.QOI"
magick "$source_dir/mouse.png" -trim +repage -resize '25x30>' \
    -gravity center -background none -extent 28x32 "$output/MOUSE.QOI"
magick -background none "$source_dir/send.svg" -trim +repage \
    -resize '30x30>' -gravity center -background none -extent 36x36 \
    "$output/SEND.QOI"

echo "Prepared standard QOI assets in $output"
