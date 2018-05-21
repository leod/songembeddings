import json

def load(filename):
    with open(filename) as f_config:
        return json.load(f_config)
