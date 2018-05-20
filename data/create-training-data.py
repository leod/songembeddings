#!/usr/bin/env python

import sys
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Create training data from a list of songs in a library. The song information is passed line-by-line on STDIN.')
parser.add_argument('--library', help='Path to the library in which the songs are stored')
parser.add_argument('--out', help='Name of the directory dedicated to the training data')
parser.add_argument('--no_shuffle', action='store_true', help='Whether to shuffle the training data (on the level of examples)')
parser.add_argument('--sequence_length', type=int, help='Sequence length of one example')
args = parser.parse_args()

if os.path.isdir(args.out):
    sys.stderr.write("Output directory already exists: '{}'. Aborting.".format(args.out))
    sys.exit(1)
else:
    os.mkdir(args.out)

songs = []

for line in sys.stdin:
    toks = line.strip().split('\t')
    artist = toks[0]
    title = toks[1]
    features_file = os.path.join(args.library, toks[4])

    if not os.path.exists(features_file):
        sys.stderr.write("WARNING: Ignoring nonexistent file '{}'\n".format(features_file))
        continue

    songs.append((artist, title, features_file))

songs_path = os.path.join(args.out, 'SONGS')
print("Writing song list to '{}'", songs_path)

with open(songs_path, 'w') as f_songs:
    for song in songs:
        f_songs.write('{}\t{}\n'.format(song[0], song[1]))

song_examples = []

print("Reading in songs from library '{}' ...".format(args.library))

for song in songs:
    features = np.transpose(np.load(song[2]))
    length = np.shape(features)[0]
    num_examples = length // args.sequence_length

    features_prefix = features[:num_examples * args.sequence_length, :] 
    examples = np.split(features_prefix, num_examples)

    print('{} - {}: {}'.format(song[0], song[1], np.shape(examples)))

    song_examples.append(examples)

all_examples = np.vstack(song_examples)

print('All training examples: {}'.format(np.shape(all_examples)))

examples_path = os.path.join(args.out, 'examples.npy')
print("Writing training examples to '{}'".format(examples_path))

np.save(examples_path, all_examples)

