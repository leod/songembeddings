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
    album=$(echo "$probe" | grep "album           :" | head -n1 | cut -d':' -f2 | sed -e 's/^ //')
    title=$(echo "$probe" | grep "title           :" | head -n1 | cut -d':' -f2 | sed -e 's/^ //')
    name=$(echo "$artist - $album - $title" | sed -e 's/[^A-Za-z0-9._-]/_/g')

    new_file=$name.mp3
    features_file=$name.features.npy

    if [ -e "$library/$new_file" ] && [ -e "$library/$features_file" ]
    then
        h=$(md5sum < "$file")
        h_new=$(md5sum < "$library/$new_file")
        if [ "$h" = "$h_new" ] 
        then
            echo "Already have '$file', skipping"
            continue
        fi
    fi

    entry="$artist\t$album\t$title\t$file\t$new_file\t$features_file" 
    echo -e $entry

    cp "$file" "$library"/"$new_file"

    "$(dirname $0)"/extract-features.py --file "$file" --out "$library"/"$features_file"

    echo -e $entry >> $library/ENTRIES
done
