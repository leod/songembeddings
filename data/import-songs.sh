#!/bin/bash

library=$1
args=${@:2}

echo "Additional arguments for extract-features.py: $args"
echo "Importing to library: '$library' ..."

mkdir -p "$library"

while read file
do
    probe=$(ffprobe "$file" 2>&1)
    artist=$(echo "$probe" | grep "artist          :" | head -n1 | cut -d':' -f2 | sed -e 's/^ //')
    title=$(echo "$probe" | grep "title           :" | head -n1 | cut -d':' -f2 | sed -e 's/^ //')
    name="$artist - $title"

    new_file=$name.mp3
    features_file=$name.features.npy

    entry="$artist\t$title\t$file\t$new_file\t$features_file" 
    echo -e $entry

    cp "$file" "$library"/"$new_file"

    "$(dirname $0)"/extract-features.py --file "$file" --out "$library"/"$features_file"

    echo -e $entry >> $library/ENTRIES
done
