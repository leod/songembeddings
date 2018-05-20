#!/bin/bash

library=$1

while read artist
do
    find "$library"/"$artist" -name '*.mp3'
done
