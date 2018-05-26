#!/usr/bin/env python

import argparse

import numpy as np
import tensorflow as tf

import config
import data
import model

parser = argparse.ArgumentParser(description='Save learned embeddings in a file.')
parser.add_argument('--config', '-c', required=True, help='Config file')
parser.add_argument('--data', '-d', required=True, help='Training data directory')
parser.add_argument('--ckpt', required=True, help='TensorFlow checkpoint file')
parser.add_argument('--out', '-o', required=True, help='File to save embeddings in (numpy format)')
args = parser.parse_args()

config = config.load(args.config)
train_songs = data.load_songs_info(args.data)

input_song_ids = tf.placeholder(tf.int32, [None])
target_feature_sequences = tf.placeholder(
    tf.float32,
    [None, config['sequence_length'], config['num_features']],
)

feature_outputs = model.build(config, len(train_songs), input_song_ids, target_feature_sequences)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, args.ckpt)

    print('Model restored.')

    with tf.variable_scope("", reuse=True):
        song_embedding_table = tf.get_variable('song_embedding_table')

    embeddings = sess.run(song_embedding_table)
    print("Saving embeddings of shape {} to file '{}'".format(np.shape(embeddings), args.out))
    np.save(args.out, embeddings)
