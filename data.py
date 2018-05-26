import pickle
import os

def load_songs_info(path):
    songs_path = os.path.join(path, 'songs')
    print("Loading song list from '{}' ...".format(songs_path))

    songs = []

    with open(songs_path) as f_songs:
        for line in f_songs:
            songs.append(line.strip())

    print("Got {} songs".format(len(songs)))

    return songs

def load_examples(path):
    examples_path = os.path.join(path, 'examples.pkl')
    print("Loading examples from '{}' ...".format(examples_path))

    with open(examples_path, 'rb') as f_examples:
        examples = pickle.load(f_examples)

    print("Got {} examples".format(len(examples)))

    return examples

def load(path):
    songs = load_songs_info(path)
    examples = load_examples(path)

    return (songs, examples)
