#!/usr/bin/env python

import os
import csv
import numpy        as np
import tensorflow   as tf
import deepscite.model as model
import deepscite.utils as utils

flags = tf.app.flags

# Training
flags.DEFINE_integer("iterations",          2000,   "Number of total training iterations.")
flags.DEFINE_integer("minibatch_size",      1000,   "Number of word vectors we should optimise over at each iteration step.")
flags.DEFINE_float("learning_rate",         1e-2,   "Learning rate.")
flags.DEFINE_integer("embedded_word_size",  250,    "Dimension of the word embedding.")
flags.DEFINE_integer("conv_size",           3,      "Number of words the convolution looks at.")
flags.DEFINE_integer("conv_stride",         1,      "The stride of the convolution.")
flags.DEFINE_integer("conv_features",       1,      "Per word, the number of features to emit in a new vector.")
flags.DEFINE_float("weights_reg_scale",     1e-6,   "Scale for the weights.")
flags.DEFINE_float("activity_reg_scale",    1e-6,   "Scale for the activities.")
flags.DEFINE_float("embedding_reg_scale",   1e-6,   "Scale for the word embedding.")

# Data
flags.DEFINE_integer("word_vector_size",    500,    "The maximum number of words we ever want to be able to process.")
flags.DEFINE_string("data_dir",             None,   "The location of your datasets. Should contain `train.csv`, `test.csv` and `validation.csv`.")

# Meta
flags.DEFINE_integer("report_frequency",        10,     "The number of times to write status reports.")
flags.DEFINE_integer("validation_frequency",    10,     "The number of times to perform validaiton.")
flags.DEFINE_integer("checkpoint_frequency",    10,     "The number of times to write checkpoints.")
flags.DEFINE_string("log_path",                 None,   "A temporary to save checkpoints into.")
flags.DEFINE_string("save_path",                None,   "A place to save the final checkpoint into.")
flags.DEFINE_integer("seed",                    None,   "A random seed so that we can get consistent output.")
flags.DEFINE_boolean("reuse_checkpoints",       False,  "True if we should re-use existing checkpoints; false otherwise.")

conf = flags.FLAGS


def validate_params():
    if conf.seed:
        tf.set_random_seed(conf.seed)
        np.random.seed(conf.seed)

    if conf.data_dir is None:
        print("`data_dir` parameter is required.")
        exit(1)

    if conf.log_path is None:
        print("`log_path` parameter is required.")
        exit(3)


def main(_):
    validate_params()
    train()


def load_file(filename):
    """ Expects a CSV file like:

        id,wordset_1_ids,wordset_2_ids,probability
        1,981723 12938 123912 423413 2 1,91 2894 123 195,1

    We just read the whole file in and return it as a list
    of tuples.
    """
    rows = []
    with open(os.path.join(conf.data_dir, filename), "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    return np.array(rows)


def get_datapoints(dataset):
    X1     = np.zeros([conf.minibatch_size, conf.word_vector_size, 1],   dtype=np.int32)
    X2     = np.zeros([conf.minibatch_size, conf.word_vector_size, 1],   dtype=np.int32)
    Y      = np.empty([conf.minibatch_size],                             dtype=np.float32)
    masks1 = np.zeros([conf.minibatch_size, conf.word_vector_size, 1],   dtype=np.int32)
    masks2 = np.zeros([conf.minibatch_size, conf.word_vector_size, 1],   dtype=np.int32)
    sizes1 = np.empty([conf.minibatch_size],                             dtype=np.float32)
    sizes2 = np.empty([conf.minibatch_size],                             dtype=np.float32)

    for k, d in enumerate(dataset):
        # Only pick out the probability if it's in there. In the case that
        # we're evaluating on data *without* a scite probability.
        if "probability" in d:
            Y[k] = float(d["probability"])

        wordset_1_ids   = d["wordset_1_ids"].split(" ")[:conf.word_vector_size]
        wordset_2_ids   = d["wordset_2_ids"].split(" ")[:conf.word_vector_size]

        for j, wid in enumerate(wordset_1_ids):
            X1[k, j]        = int(wid)
            masks1[k, j]    = 1

        for j, wid in enumerate(wordset_2_ids):
            X2[k, j]        = int(wid)
            masks2[k, j]    = 1

        sizes1[k] = len(wordset_1_ids)
        sizes2[k] = len(wordset_2_ids)

    return X1, X2, Y, masks1, masks2, sizes1, sizes2, dataset


def get_random_datapoints(dataset):
    """ Build an tensorflow input package from the given set of data. """

    points = len(dataset)
    assert conf.minibatch_size <= points, \
                "Dataset is too small: {}. Try reducing your minibatch size: {}.".format(points, conf.minibatch_size)

    perm   = np.random.permutation(points)
    subset = dataset[perm][:conf.minibatch_size]

    return get_datapoints(subset)


def train():
    train_data      = load_file("train.csv")
    validation_data = load_file("validation.csv")

    m = model.JointEmbeddingModelForBinaryClassification(conf.embedded_word_size)

    checkpoint_base_path = conf.log_path + "/checkpoint"

    with tf.Session() as sess:
        model_params = m.graph(
                conf.minibatch_size,
                utils.get_vocab_size(conf.data_dir),
                conf.word_vector_size,
                conf.conv_size,
                conf.conv_stride,
                conf.conv_features,
                conf.weights_reg_scale,
                conf.activity_reg_scale,
                conf.embedding_reg_scale
                )

        optimiser = tf.train.AdamOptimizer(conf.learning_rate)
        train_op  = optimiser.minimize(model_params.loss, var_list=tf.trainable_variables())

        if not os.path.exists(conf.log_path):
            os.makedirs(conf.log_path)

        writer = tf.train.SummaryWriter(conf.log_path, sess.graph)
        saver  = tf.train.Saver()

        latest_checkpoint = tf.train.latest_checkpoint(conf.log_path)

        if conf.reuse_checkpoints and latest_checkpoint is not None:
            print("Restoring checkpoint...: " + latest_checkpoint)
            saver.restore(sess, latest_checkpoint)
            starting_iteration = int(latest_checkpoint.split('-')[-1]) + 1
        else:
            print("Initialising new model...")
            sess.run(tf.initialize_all_variables())
            starting_iteration = 0

        summary_op = tf.merge_all_summaries()

        for i in range(starting_iteration, conf.iterations + 1):

            X1, X2, Y, M1, M2, S1, S2, _ = get_random_datapoints(train_data)
            data = {model_params.wordset_1: X1,
                    model_params.wordset_2: X2,
                    model_params.probs: Y,
                    model_params.wordset_1_masks: M1,
                    model_params.wordset_2_masks: M2,
                    model_params.wordset_1_lengths: S1,
                    model_params.wordset_2_lengths: S2}

            _, summary_value, alpha, loss_value = sess.run([train_op, summary_op, model_params.alpha, model_params.loss], feed_dict=data)

            writer.add_summary(summary_value, i)

            if i % conf.report_frequency == 0:
                print("Iteration #{}, Loss: {}, α: {}.".format(i, loss_value, alpha))


            if i % conf.checkpoint_frequency == 0:
                checkpoint_path = saver.save(sess, checkpoint_base_path, global_step=i)
                print("Checkpointed: {}.".format(checkpoint_path))


            if i % conf.validation_frequency == 0:
                X1, X2, Y, M1, M2, S1, S2, _ = get_random_datapoints(validation_data)
                data = {model_params.wordset_1: X1,
                        model_params.wordset_2: X2,
                        model_params.probs: Y,
                        model_params.wordset_1_masks: M1,
                        model_params.wordset_2_masks: M2,
                        model_params.wordset_1_lengths: S1,
                        model_params.wordset_2_lengths: S2}

                accuracy_value = sess.run(model_params.accuracy, feed_dict=data)

                print("Iteration #{}, Validation-set accuracy: {}.".format(i, accuracy_value))

        if not os.path.exists(conf.save_path):
            os.makedirs(conf.save_path)
            
        saver.save(sess, conf.save_path + "/last_checkpoint")
