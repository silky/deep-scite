import os
import re
import nltk

def load_vocabulary(base_dir):
    with open(os.path.join(base_dir, "vocab.txt"), "r", encoding="utf-8") as f:
        vocab_list = f.read().splitlines()
    return vocab_list

def get_vocab_size(base_dir):
    with open(os.path.join(base_dir, "vocab.txt"), "r", encoding="utf-8") as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def to_words(title):
    title = title.lower()

    # Let's filter out maths inside "$". It's important that
    # we do this in a non-greedy way.
    title = re.sub(r"(\$.*?\$)", "", title)
    words = nltk.word_tokenize(title)

    return words



