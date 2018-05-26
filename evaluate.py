#!/usr/bin/python -u

import argparse

import numpy as np
import scipy.spatial

import data

parser = argparse.ArgumentParser(description='Evaluate embeddings somehow...')
parser.add_argument('--data', '-d', required=True, help='Training data directory')
parser.add_argument('--file', '-f', required=True, help='Embeddings file (numpy format)')
args = parser.parse_args()

train_songs = data.load_songs_info(args.data)
embeddings = np.load(args.file)

print("Loaded embeddings of shape {} from file '{}'".format(np.shape(embeddings), args.file))

def album_distance(embeddings, songs1, songs2):
    x = 0
    for song1 in songs1:
        for song2 in songs2:
            x += scipy.spatial.distance.cosine(embeddings[song1], embeddings[song2])
    return x / (len(songs1) * len(songs2))

albums = {}

longest_artist = 0
longest_album = 0

for i, song in enumerate(train_songs):
    albums.setdefault((song[0], song[1]), []).append(i)

    longest_artist = max(longest_artist, len(song[0]))
    longest_album = max(longest_album, len(song[1]))

album_format = '{: <' + str(longest_artist)  + '} - {: <' + str(longest_album) + '}'

print('Got {} albums'.format(len(albums)))

for (artist1, album1), songs1 in sorted(albums.items(), key=lambda s: s[0][0] + ' - ' + s[0][1]):
    closest_album = None
    closest_dist = None

    for ((artist2, album2), songs2) in albums.items():
        if artist1 == artist2 and album1 == album2:
            continue

        dist = album_distance(embeddings, songs1, songs2)
        if closest_album is None or dist < closest_dist:
            closest_album = (artist2, album2)
            closest_dist = dist

    print((album_format + ': ' + album_format + ': {}').format(artist1, album1, closest_album[0], closest_album[1], dist))
