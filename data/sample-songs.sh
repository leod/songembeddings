#!/bin/bash

find "$1" -name '*.mp3' | \
    shuf | \
    head -n $2 | \
    while read file
    do
        duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")
        if [ 1 -eq "$(echo "$duration > 120.0" | bc)" ]
        then
            if [ 1 -eq "$(echo "$duration < 300.0" | bc)" ]
            then
                echo $file $duration
            fi
        fi
    done | \
        head -n $3
