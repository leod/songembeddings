#!/usr/bin/env python

import argparse

import numpy as np

import config
import data
import model

parser = argparse.ArgumentParser(description='Create baseline pseudo-embeddings by averaging the features of a song.')
parser.add_argument('--data', '-d', required=True, help='Training data directory')
parser.add_argument('--out', '-o', required=True, help='File to save embeddings in (numpy format)')
args = parser.parse_args()

train_songs, train_examples = data.load(args.data)

num_features = train_examples[0][1].shape[1]
print('Number of features: {}'.format(num_features))

embeddings = np.zeros((len(train_songs), num_features))
num_samples = np.zeros(len(train_songs))

for (song_id, feature_sequence) in train_examples:
    x = np.mean(feature_sequence, axis=0)
    embeddings[song_id] += x
    num_samples[song_id] += 1

embeddings /= num_samples[:, None]

print("Saving embeddings of shape {} to file '{}'".format(np.shape(embeddings), args.out))
np.save(args.out, embeddings)
