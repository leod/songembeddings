#!/usr/bin/env python

import librosa
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Extract features of a song')
parser.add_argument('--file', help='File of the song')
parser.add_argument('--out', help='File to store the output in')
parser.add_argument('--hop_length', type=int, default=512, help='MFCC hop length')
parser.add_argument('--n_mfcc', type=int, default=13, help='Number of MFCCs')
args = parser.parse_args()

y, sr = librosa.load(args.file)
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=args.hop_length, n_mfcc=args.n_mfcc)
np.save(args.out, mfcc)
