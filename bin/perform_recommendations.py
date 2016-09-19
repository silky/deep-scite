#!/usr/bin/env python

"""
    Given a CSV file with:

        arxiv_id,wordset_1_ids,wordset_2_ids

    we'll emit a CSV like:

        arxiv_id,scite_prob,title,abstract,title_goodness_vals,abstract_goodness_vals

"""

import os
import csv
import deepscite.model as model
import deepscite.train as train
import deepscite.utils as utils
import numpy        as np
import tensorflow   as tf

flags = tf.app.flags

flags.DEFINE_string("input_csv",        None, "The csv file containing the items to consider recommending.")
flags.DEFINE_string("output_csv",       None, "The csv file containing our commendations.")
flags.DEFINE_string("checkpoint_path",  None, "Location of the latest checkpoint.")
flags.DEFINE_string("report_file",      None, "HTML report file.")

conf = tf.app.flags.FLAGS


def main(_):
    eval_data   = train.load_file(conf.input_csv)
    vocab_list  = utils.load_vocabulary(conf.data_dir)

    # ooooh.
    conf.minibatch_size = len(eval_data)

    m = model.JointEmbeddingModelForBinaryClassification(conf.embedded_word_size)

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

        saver  = tf.train.Saver()
        latest_checkpoint = tf.train.latest_checkpoint(conf.checkpoint_path)
        
        saver.restore(sess, latest_checkpoint)

        X1, X2, _, M1, M2, S1, S2, subset = train.get_datapoints(eval_data)
        data = {model_params.wordset_1: X1,
                model_params.wordset_2: X2,
                model_params.wordset_1_masks: M1,
                model_params.wordset_2_masks: M2,
                model_params.wordset_1_lengths: S1,
                model_params.wordset_2_lengths: S2}

        # Calculate the recommendations
        set1_activations, set2_activations, final_probs = sess.run([
            tf.squeeze(model_params.conv_wordset_1_activity, [2,3]),
            tf.squeeze(model_params.conv_wordset_2_activity, [2,3]),
            model_params.final_probs], 
            feed_dict=data)

        points = [eval_data, final_probs, set1_activations, set2_activations]

        # Write the CSV
        fields = ["arxiv_id", "scite_probability", "title", "abstract", 
                    "title_word_goodness", "abstract_word_goodness"]

        output = []
        header = """
<style>
.good { color: blue; }
.bad  { color: red;  }

span.agree { background: green; }
span.disagree  { background: red; }
</style>
        """
        output.append( header )

        with open(os.path.join(conf.data_dir, conf.output_csv), "w") as csvfile:
            w = csv.DictWriter(csvfile, fieldnames=fields)
            w.writeheader()

            for (d, scite_prob, title_activations, abstract_activations) in zip(*points):
                title_words     = [vocab_list[int(word_id)] for word_id in d["wordset_1_ids"].split(" ")]
                title_words     = title_words[:conf.word_vector_size]

                abstract_words  = [vocab_list[int(word_id)] for word_id in d["wordset_2_ids"].split(" ")]
                abstract_words  = abstract_words[:conf.word_vector_size]
                #
                title_html      = words_to_html(title_words,    title_activations)
                abstract_html   = words_to_html(abstract_words, abstract_activations)

                w.writerow({"arxiv_id":        d["id"],
                    "scite_probability":        scite_prob,
                    "title":                    " ".join(title_words),
                    "abstract":                 " ".join(abstract_words),
                    "title_word_goodness":      " ".join(map(str, title_activations)),
                    "abstract_word_goodness":   " ".join(map(str, abstract_activations))})
                #
                agreement = "disagree"
                if scite_prob > 0.8:
                    decision = "scite"
                    if int(d["probability"]) == 1:
                        agreement = "agree"
                else:
                    decision = "ignore"
                    if int(d["probability"]) == 0:
                        agreement = "agree"
                #
                html = """
                    <p><em>[{}]</em> - <strong>{}</strong> prob to scite: {}</small>
                        <br /><br />
                        {}
                        <br />
                        <small><span class='{}'>decision: {}</span></small>
                    </p>
                """.format(d["id"], title_html, str(round(scite_prob, 2)), abstract_html, 
                        agreement, decision)
                output.append( html )
            #
        #
        #
        with open(os.path.join(conf.data_dir, conf.report_file), "w") as f:
            f.write( "\n".join(output) )


                   
def words_to_html(words, activations):
    good_words = []
    bad_words  = []

    elts = []

    threshold = 5
    for k, w in enumerate(words):
        # activation = round(sum(activations[j] for j in [k-1, k, k+1] if j >= 0 and j < len(activations)), 2)
        activation = round(float(activations[k]), 2)

        class_ = "neutral"
        if activation > threshold:
            good_words.append(w)
            class_ = "good"

        if activation < -threshold:
            bad_words.append(w)
            class_ = "bad"

        elts.append("<span class='{}' title='({},{})'>{}</span>".format(class_, activation,
                    round(float(activations[k]), 2), w))
    
    return " ".join(elts)

if __name__ == "__main__":
    tf.app.run()
