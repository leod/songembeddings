#!/usr/bin/env python

import sys
import os
import argparse
import numpy as np
import pickle
import random

parser = argparse.ArgumentParser(description='Create training data from a list of songs in a library. The song information is passed line-by-line on STDIN.')
parser.add_argument('--library', required=True, help='Path to the library in which the songs are stored')
parser.add_argument('--out', required=True, help='Name of the directory dedicated to the training data')
parser.add_argument('--sequence_length', type=int, required=True, help='Sequence length of one example')
parser.add_argument('--no_shuffle', action='store_true', help='Whether to shuffle the training data (on the level of examples)')
args = parser.parse_args()

if os.path.isdir(args.out):
    sys.stderr.write("Output directory already exists: '{}'. Aborting.\n".format(args.out))
    sys.exit(1)
else:
    os.mkdir(args.out)

songs = []

for line in sys.stdin:
    toks = line.strip().split('\t')
    artist = toks[0]
    album = toks[1]
    title = toks[2]
    features_file = os.path.join(args.library, toks[5])

    if not os.path.exists(features_file):
        sys.stderr.write("WARNING: Ignoring nonexistent file '{}'\n".format(features_file))
        continue

    songs.append((artist, album, title, features_file))

songs_path = os.path.join(args.out, 'songs')
print("Writing song list to '{}'".format(songs_path))

with open(songs_path, 'w') as f_songs:
    for song in songs:
        f_songs.write('{}\t{}\t{}\n'.format(song[0], song[1], song[2]))

song_examples = []

print("Reading in songs from library '{}' ...".format(args.library))

for (song_id, song) in enumerate(songs):
    features = np.transpose(np.load(song[3]))
    length = np.shape(features)[0]
    num_examples = length // args.sequence_length

    features_prefix = features[:num_examples * args.sequence_length, :] 
    examples = np.split(features_prefix, num_examples)

    print('Song: {} - {} - {}: {}'.format(song[0], song[1], song[2], np.shape(examples)))

    for example in examples:
        song_examples.append((song_id, example))

print('Got {} training examples'.format(len(song_examples)))
print('Average number of examples per song: {:.2f}'.format(len(song_examples) / len(songs)))

if not args.no_shuffle:
    print("Shuffling examples")
    random.shuffle(song_examples)

examples_path = os.path.join(args.out, 'examples.pkl')
print("Writing training examples to '{}'".format(examples_path))

with open(examples_path, 'wb') as f_examples:
    pickle.dump(song_examples, f_examples)

#np.save(examples_path, song_examples)
