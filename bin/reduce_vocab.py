#!/usr/bin/env python

import os
import csv
import tensorflow       as tf
import deepscite.train  as t
import deepscite.utils  as utils
import itertools

from collections import defaultdict

flags = tf.app.flags

# Data_dir already provided by 'train'
flags.DEFINE_string("new_data_dir",         None,   "The location of the new datasets.")
flags.DEFINE_integer("required_word_count", 1,      "Limit words used to those that appear k-times or more.")

conf = flags.FLAGS

# Original WordID -> Count
word_count_dict = defaultdict(int)

# Original WordID -> New WordID
remapping_dict  = {}

def main(_):
    if conf.data_dir is None:
        print("`data_dir` parameter is required.")
        exit(1)

    if conf.new_data_dir is None:
        print("`new_data_dir` parameter is required.")
        exit(1)

    train_data      = t.load_file("train.csv")
    validation_data = t.load_file("validation.csv")
    test_data       = t.load_file("test.csv")
    
    # Work out word counts
    for d in itertools.chain(train_data, validation_data, test_data):
        wids  = t.wordstring_to_wids(d["wordset_1_ids"])
        wids += t.wordstring_to_wids(d["wordset_2_ids"])

        for w in wids:
            word_count_dict[w] += 1

    vocab_size = 0
    original_vocab = utils.load_vocabulary(conf.data_dir)

    if not os.path.exists(conf.new_data_dir):
        os.makedirs(conf.new_data_dir)

    # Re-write the vocab file, cou the vocab, assign indexes
    with open(os.path.join(conf.new_data_dir, "vocab.txt"), "w") as f:
        for (k, v) in word_count_dict.items():
            if v >= conf.required_word_count:
                remapping_dict[k] = vocab_size
                vocab_size += 1
                f.write(original_vocab[k] + "\n")


    data = [ ("train.csv", train_data),
             ("validation.csv", validation_data),
             ("test.csv", test_data) ]

    # Re-write dataset so that we only have rows that actually have words.
    for (filename, dataset) in data:
        with open(os.path.join(conf.new_data_dir, filename), "w") as csvfile:
            fields = dataset[0].keys()
            w = csv.DictWriter(csvfile, fieldnames=fields)
            w.writeheader()
            for d in only_with_words(dataset):
                w.writerow(d)

    print("New vocab size: " + str(vocab_size))


def only_with_words (dataset):
    for d in dataset:
        wids1 = t.wordstring_to_wids(d["wordset_1_ids"])
        wids2 = t.wordstring_to_wids(d["wordset_2_ids"])

        wids1 = [remapping_dict[w] for w in wids1 if word_count_dict[w] >= conf.required_word_count]
        wids2 = [remapping_dict[w] for w in wids2 if word_count_dict[w] >= conf.required_word_count]

        if len(wids1) == 0 or len(wids2) == 0:
            continue
        else:
            d["wordset_1_ids"] = " ".join(map(str, wids1))
            d["wordset_2_ids"] = " ".join(map(str, wids2))
            yield d


if __name__ == "__main__":
    tf.app.run()

