#!/usr/bin/env python

import argparse
import pickle
import os
import pprint
import sys
import time

import tensorflow as tf

config = {}
config['sequence_length'] = 50
config['num_features'] = 13
config['hidden_size'] = 128
config['embedding_size'] = 64
config['batch_size'] = 100
config['learning_rate'] = 0.001
config['max_steps'] = 1000000

parser = argparse.ArgumentParser(description='Train song embeddings.')
parser.add_argument('--data', required=True, help='Training data directory')
args = parser.parse_args()

songs_path = os.path.join(args.data, 'songs')
print("Loading song list from '{}' ...".format(songs_path))

with open(songs_path) as f_songs:
    songs = []
    for line in f_songs:
        toks = line.strip().split('\t')
        songs.append((toks[0], toks[1]))

print("Got {} songs".format(len(songs)))

examples_path = os.path.join(args.data, 'examples.pkl')
print("Loading training examples from '{}' ...".format(examples_path))

with open(examples_path, 'rb') as f_examples:
    train_examples = pickle.load(f_examples)

print("Got {} examples".format(len(train_examples)))

print("Model configuration: ")
pprint.PrettyPrinter(indent=4).pprint(config)

input_song_ids = tf.placeholder(tf.int32, [None])
target_feature_sequences = tf.placeholder(
    tf.float32,
    [None, config['sequence_length'], config['num_features']],
)

with tf.device("/cpu:0"):
    song_embedding_table = tf.get_variable(
        "song_embedding_table",
        [len(songs), config['embedding_size']],
    )

    # [batch_size, embedding_size]
    song_embeddings = tf.nn.embedding_lookup(song_embedding_table, input_song_ids)

decoder_rnn = tf.contrib.rnn.BasicLSTMCell(config['hidden_size'])
decoder_initial_state = decoder_rnn.zero_state(tf.shape(input_song_ids)[0], tf.float32)

# [batch_size, sequence_length, embedding_size]
tiled_song_embeddings = tf.tile(
    tf.expand_dims(song_embeddings, 1),
    [1, config['sequence_length'], 1],
)

# [batch_size, sequence_length, embedding_size+num_features]
decoder_inputs = tf.concat([tiled_song_embeddings, target_feature_sequences], axis=2)

decoder_outputs, _decoder_state = tf.nn.dynamic_rnn(
    decoder_rnn,
    decoder_inputs,
    initial_state=decoder_initial_state,
)

output_W = tf.get_variable('output_W', [config['hidden_size'], config['num_features']])
tiled_output_W = tf.tile(
    tf.expand_dims(output_W, 0),
    [config['batch_size'], 1, 1],
)

output_b = tf.get_variable('output_b', [config['num_features']])
feature_outputs = tf.matmul(decoder_outputs, tiled_output_W) + output_b

loss = tf.losses.mean_squared_error(target_feature_sequences, feature_outputs)

tf.summary.scalar('loss', loss)

optimizer = tf.train.AdamOptimizer(config['learning_rate'])

global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)

summary = tf.summary.merge_all()

saver = tf.train.Saver()

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('logs', sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)

    next_data_index = 0
    num_seen_examples = 0

    for step in range(config['max_steps']):
        start_time = time.time()

        song_ids = []
        feature_sequences = []

        while len(song_ids) < config['batch_size']:
            song_ids.append(train_examples[next_data_index][0])
            feature_sequences.append(train_examples[next_data_index][1])
            next_data_index += 1
            num_seen_examples += 1
            if next_data_index == len(train_examples):
                next_data_index = 0

        feed_dict = {
            input_song_ids: song_ids,
            target_feature_sequences: feature_sequences,
        }
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

        duration = time.time() - start_time

        if step % 100 == 0:
            print('Step {} ({:.2} epochs): loss={:.2} ({:.3} sec)'.format(step, num_seen_examples / len(train_examples), loss_value, duration))

            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        if (step + 1) % 1000 == 0 or (step + 1) == config['max_steps']:
            saver.save(sess, './model.ckpt', global_step=step)
