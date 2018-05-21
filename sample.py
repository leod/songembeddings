#!/usr/bin/env python

import argparse
import sys

import numpy as np
import tensorflow as tf
import librosa

import config
import model

from IPython.lib.display import Audio

parser = argparse.ArgumentParser(description='Train song embeddings.')
parser.add_argument('--config', '-c', required=True, help='Config file')
parser.add_argument('--ckpt', required=True, help='TensorFlow checkpoint file')
parser.add_argument('--song_id', required=True, type=int, help='ID of the song to sample')
parser.add_argument('--n_samples', type=int, default=100, help='Number of sequential samples to take')
args = parser.parse_args()

config = config.load(args.config)

input_song_ids = tf.placeholder(tf.int32, [None])
target_feature_sequences = tf.placeholder(
    tf.float32,
    [None, None, config['num_features']],
)
feature_outputs = model.build(config, 382, input_song_ids, target_feature_sequences)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, args.ckpt)

    print('Model restored.')

    outputs = [np.zeros((1, 1, config['num_features']))]

    # This is super inefficient since it does not use the known hidden states
    # and instead recomputes from scratch
    for i in range(args.n_samples):
        if (i + 1) % 50 == 0:
            #print(outputs[-1])
            sys.stdout.write('.')
            sys.stdout.flush()

        history = np.concatenate(outputs, axis=1)

        feed_dict = {
            input_song_ids: [args.song_id],
            target_feature_sequences: history,
        }

        new_outputs = sess.run(feature_outputs, feed_dict=feed_dict)
        last_output = np.expand_dims(new_outputs[:, -1, :], axis=1)
        outputs.append(last_output)

    sys.stdout.write('\n')

    def invlogamplitude(S):
        """librosa.logamplitude is actually 10_log10, so invert that."""
        return 10.0*(S/10.0)

    # Reconstruct audio:
    # https://github.com/librosa/librosa/issues/424

    mfccs = np.transpose(np.squeeze(np.concatenate(outputs, axis=1), 0))
    n_mfcc = mfccs.shape[0]
    n_mel = 128
    dctm = librosa.filters.dct(n_mfcc, n_mel)
    n_fft = 2048
    sr = 22050
    mel_basis = librosa.filters.mel(sr, n_fft)
    bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis), axis=0))
    recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T, invlogamplitude(np.dot(dctm.T, mfccs)))
    y_len = int(sr * 2.325)
    excitation = np.random.randn(y_len)
    E = librosa.stft(excitation)
    print(np.shape(recon_stft))
    print(np.shape(excitation))
    print(np.shape(E))
    print(recon_stft)
    recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))

    Audio(recon, rate=sr)
