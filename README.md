# `DeepScite` - A Simple Convolutional-based Recommendation Model

# Installation

1. Clone this repository:

  ```
  git clone https://github.com/silky/deep-scite
  ```

2. Use [conda](http://conda.pydata.org/docs/download.html) or (virtualenv) and
   create an environment that has Python 3.5.

   `conda create -n deep-scite python=3.5`

3. Install TensorFlow (version `0.10`):

  Head over
  [here](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#using-pip)
  and pick the version that is appropriate for your architecture.

4. Install the requirements

  `pip install -r requirements.txt`

5. Install `nltk` language packs

  In order to tokenise strings, we use the `nltk` package. It requires
  us to download some data before using it though. To do so, run:

  ````
  python -c 'import nltk; nltk.download("punkt")'
  ````

6. Install this library in `develop` mode

  `python setup.py develop`


# Usage

From the root directory of this project:

1. Activate the `deep-scite` environment

  `source activate deep-scite`

2. Train the model on the `noon` data set, and emit recommendations

  `./bin/run_model.py`

  This will run through the steps defined in `model.yaml`.

3. Open up `./data/noon/report.html` in your browser and observe recommendations.
