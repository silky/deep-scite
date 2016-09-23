"""
    Looks at (title, abstract) pairs and maps them to a "probability to
    scite".

    - "wordset_1" is used for the titles,
    - "wordset_2" is used for the abstracts.

    There is a little bit of busywork involved in making sure the model
    doesn't preference those titles/abstracts that are shorter. To this end we
    need to introduce "masks" and "lengths" of the respective inputs, and use
    these to kill off any contributions from empty word positions.
"""

import tensorflow as tf
from collections import namedtuple

ModelParameters = namedtuple("ModelParameters",
        ["wordset_1", "wordset_2", "probs", "wordset_1_masks",
            "wordset_2_masks", "wordset_1_lengths", "wordset_2_lengths",
            "conv_wordset_1_activity", "conv_wordset_2_activity", 
            "loss",
            "accuracy",
            "conv_wordset_1_weights", "conv_wordset_2_weights",
            "joint_means", "final_probs",
            "alpha"
            ])

class JointEmbeddingModelForBinaryClassification():
    """
        Allows one to send two word vectors (say titles and abstracts) and
        then learns how to classify these two bits of information at once,
        with a weighting parameter.
    """
    def __init__(self, embedded_word_size=250, init_stddev=1e-4):
        self.embedded_word_size = embedded_word_size
        self.init_stddev        = init_stddev

 
    def _initialise(self, shape):
        return tf.truncated_normal(shape, stddev=self.init_stddev, dtype=tf.float32)


    def graph(self, minibatch_size, vocab_size, input_size, 
                conv_size=3, conv_stride=1, conv_features=1,
                weights_reg_scale=1e-6, activity_reg_scale=1e-6, embedding_reg_scale=1e-6
                ):

        wordset_1 = tf.placeholder(
                tf.int32,
                shape = (minibatch_size, input_size, 1),
                name  = "wordset_1_indicies"
                )

        wordset_2 = tf.placeholder(
                tf.int32,
                shape = (minibatch_size, input_size, 1),
                name  = "wordset_2_indicies"
                )

        # We need to build the trainable word embedding part.
        with tf.name_scope("word_vectors"):
            word_vectors_init   = self._initialise([vocab_size, self.embedded_word_size])
            word_vectors        = tf.Variable(word_vectors_init, name="word_vectors")

        tf.histogram_summary("word_embedding", word_vectors)

        wordset_1_vects = tf.nn.embedding_lookup(word_vectors, wordset_1)
        wordset_2_vects = tf.nn.embedding_lookup(word_vectors, wordset_2)

        probs = tf.placeholder(tf.float32,
                shape = (minibatch_size),
                name  = "class_a_probability"
                )

        wordset_1_masks = tf.placeholder(tf.float32,
                shape = (minibatch_size, input_size, 1),
                name  = "wordset_1_masks"
                )

        wordset_2_masks = tf.placeholder(tf.float32,
                shape = (minibatch_size, input_size, 1),
                name  = "wordset_2_masks"
                )

        wordset_1_lengths = tf.placeholder(tf.float32,
                shape = (minibatch_size),
                name  = "wordset_1_lengths"
                )

        wordset_2_lengths = tf.placeholder(tf.float32,
                shape = (minibatch_size),
                name  = "wordset_2_lengths"
                )

        # Build up the convoluion. Our convolution structure will be to slide
        # a window of size <conv_size>x1 across the plane. This let's us think about
        # relating words near to each other.
        #
        # When `conv_features` is 1 we get the same number of output features
        # as input words; this let's us think about these terms as describing
        # word "usefulness".
        
        with tf.name_scope("conv"):
            with tf.name_scope("wordset_1"):
                conv_wordset_1_init      = self._initialise([conv_size, 1, self.embedded_word_size, conv_features])
                conv_wordset_1_weights   = tf.Variable(conv_wordset_1_init, name="weights")
                conv_wordset_1_bias      = tf.Variable(tf.zeros([conv_features], dtype=tf.float32), name="bias")

                conv_wordset_1_y         = tf.nn.conv2d(wordset_1_vects, conv_wordset_1_weights, [1, conv_stride, 1, 1], padding="SAME", name="y")
                conv_wordset_1_biased    = tf.nn.bias_add(conv_wordset_1_y, conv_wordset_1_bias)
                wordset_1_masks_shaped   = tf.expand_dims(wordset_1_masks, 3)

                # Set any terms where there aren't words to zero.
                conv_wordset_1_activity  = tf.mul(conv_wordset_1_biased, wordset_1_masks_shaped)

            with tf.name_scope("wordset_2"):
                conv_wordset_2_init      = self._initialise([conv_size, 1, self.embedded_word_size, conv_features])
                conv_wordset_2_weights   = tf.Variable(conv_wordset_2_init, name="weights")
                conv_wordset_2_bias      = tf.Variable(tf.zeros([conv_features], dtype=tf.float32), name="bias")

                conv_wordset_2_y         = tf.nn.conv2d(wordset_2_vects, conv_wordset_2_weights, [1, conv_stride, 1, 1], padding="SAME", name="y")
                conv_wordset_2_biased    = tf.nn.bias_add(conv_wordset_2_y, conv_wordset_2_bias)
                wordset_2_masks_shaped   = tf.expand_dims(wordset_2_masks, 3)

                # Set any terms where there aren't words to zero.
                conv_wordset_2_activity  = tf.mul(conv_wordset_2_biased, wordset_2_masks_shaped)

        # Base combination term.
        mu = tf.Variable(0.0, name="mu")

        with tf.name_scope("loss"):
            final_means_wordset_1 = tf.div(tf.reduce_sum(conv_wordset_1_activity, reduction_indices=[1,2,3]), wordset_1_lengths)
            final_means_wordset_2 = tf.div(tf.reduce_sum(conv_wordset_2_activity, reduction_indices=[1,2,3]), wordset_2_lengths)


            embedding_reg_wordset_1     = tf.nn.l2_loss(wordset_1_vects)
            embedding_reg_wordset_2     = tf.nn.l2_loss(wordset_2_vects)
            conv_weights_reg_wordset_1  = tf.nn.l2_loss(conv_wordset_1_weights)
            conv_weights_reg_wordset_2  = tf.nn.l2_loss(conv_wordset_2_weights)
            activity_reg_wordset_1      = tf.reduce_mean(tf.div(tf.reduce_sum(tf.abs(conv_wordset_1_activity), reduction_indices=[1,2,3]), wordset_1_lengths))
            activity_reg_wordset_2      = tf.reduce_mean(tf.div(tf.reduce_sum(tf.abs(conv_wordset_2_activity), reduction_indices=[1,2,3]), wordset_2_lengths))


            # Total regularisation term.
            regs_and_scales = [ (weights_reg_scale,     conv_weights_reg_wordset_1 + conv_weights_reg_wordset_2), 
                                (activity_reg_scale,    activity_reg_wordset_1 + activity_reg_wordset_2),
                                (embedding_reg_scale,   embedding_reg_wordset_1 + embedding_reg_wordset_2)
                                ]
            regularisation = sum(a*b for (a,b) in regs_and_scales)

            
            # Combination term. It's a sigmoid of `mu` so that it is bounded
            # between 0 and 1.
            alpha = tf.sigmoid(mu, name="alpha")
            tf.scalar_summary("alpha", alpha)

            joint_means = alpha * final_means_wordset_1 + (1 - alpha) * final_means_wordset_2
            batch_loss  = tf.nn.sigmoid_cross_entropy_with_logits(joint_means, probs)

            loss = tf.reduce_mean(batch_loss) + regularisation

            tf.scalar_summary("loss", loss)


        with tf.name_scope("accuracy"):
            final_probs  = tf.sigmoid(joint_means)
            abs_diff     = tf.abs(probs - final_probs)
            right_enough = tf.equal(probs, tf.to_float(tf.greater(final_probs, 0.5)))
            accuracy     = tf.reduce_mean(tf.to_float(right_enough))
            tf.scalar_summary("accuracy", accuracy)


        return ModelParameters(wordset_1, wordset_2, probs, 
                wordset_1_masks,
                wordset_2_masks,
                wordset_1_lengths,
                wordset_2_lengths,
                conv_wordset_1_activity,
                conv_wordset_2_activity,
                loss,
                accuracy,
                conv_wordset_1_weights,
                conv_wordset_2_weights,
                joint_means,
                final_probs,
                alpha)


