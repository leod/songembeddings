#!/usr/bin/env python

import argparse
import pprint
import sys
import time

import tensorflow as tf

import config
import data
import model

parser = argparse.ArgumentParser(description='Train song embeddings.')
parser.add_argument('--config', '-c', required=True, help='Config file')
parser.add_argument('--data', '-d', required=True, help='Training data directory')
parser.add_argument('--max_steps', type=int, default=1000000, help='Number of steps to train for')
args = parser.parse_args()

config = config.load(args.config)
(train_songs, train_examples) = data.load(args.data)

print("Model configuration: ")
pprint.PrettyPrinter(indent=4).pprint(config)

input_song_ids = tf.placeholder(tf.int32, [None])
target_feature_sequences = tf.placeholder(
    tf.float32,
    [None, config['sequence_length'], config['num_features']],
)
feature_outputs = model.build(config, len(train_songs), input_song_ids, target_feature_sequences)

loss = tf.losses.mean_squared_error(target_feature_sequences, feature_outputs)

optimizer = tf.train.AdamOptimizer(config['learning_rate'])
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)

tf.summary.scalar('loss', loss)
summary = tf.summary.merge_all()

saver = tf.train.Saver()

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('logs', sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)

    next_data_index = 0
    num_seen_examples = 0

    for step in range(args.max_steps):
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

        if (step + 1) % 1000 == 0 or (step + 1) == args.max_steps:
            saver.save(sess, './model.ckpt', global_step=step)
