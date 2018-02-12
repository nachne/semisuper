# Semisuper
Semi-supervised sentence classification for oncological abstracts

# Prerequisites

## Installation via setup.py

Installation requires a Unix system with *python3*, *pip3*, and Python3 dependencies *setuptools* and *nltk* installed.

> python3 setup.py

installs additional Python dependencies with "pip --user", downloads nltk data packages and the Hallmarks of Cancer corpus, and installs GENIAtagger.

## Manual installation

(from the project root containing this file as the working directory)

* mkdir semisuper/pickles semisuper/resources/HoCCorpus semisuper/silver_standard

* Install Python dependencies (can be installed using pip3; scipy may require additional math libraries):
    * numpy==1.13.3
    * scipy==0.19.1
    * scikit-learn==0.19.1
    * pandas==0.20.3
    * Unidecode==0.4.21
    * biopython==1.70
    * nltk==3.2.4
    * https://github.com/d2207197/geniatagger-python/archive/master.zip

* For training new models, .txt files from the Hallmarks of Cancer corpus from http://www.cl.cam.ac.uk/~sb895/HoC.html must be unpacked to semisuper/resources/HoCCorpus

* Optionally, to be able to use GENIAtagger, unpack and make http://www.nactem.ac.uk/tsujii/GENIA/tagger/geniatagger-3.0.2.tar.gz in directory semisuper/resources/geniatagger-3.0.2
(GENIAtagger options are currently deactivated in *semisuper/ss_model_selection.py > preproc_param_dict()*)


# Usage

## Demo 

> python3 demo.py [<max_abstracts>]

runs a precomputed model for a simple demo.
A random sample (optional parameter *max_abstracts*, default 400) of articles is processed by a pretrained classifier. Results are ouput as *demo.html*, a simple HTML file where these articles' key sentences are highlighted, with opacity according to prediction confidence.

If *semisuper/pickles/sent_test_abstracts.pickle* does not exist, new articles for the query "cancer" are fetched from PubMed.
If *semisuper/pickles/semi_pipeline.pickle* does not exist, a new model is trained.

## Training a new model

> python3 build_corpus_and_ss_classifier 

trains a new classifier, using the latest nightly CIViC dump, and saves the resulting pipeline to *semisuper/pickles/semi_pipeline.pickle*. The silver standard corpus generated in the process is saved to *semisuper/silver_standard.tsv* (the corpus is required by *key_sentence_predictor.py* for normalising confidence scores to the [0,1] range).

Currently, this performs hyperparameter search for iterative SVM. To test additional models, the return statement of function *estimator_list()* in *semisuper/ss_model_selection.py* must be changed. Likewise, n-gram options and feature selection methods can be uncommented in the same module's *preproc_param_dict()* function. The parameter combination with the highest accuracy score in simple 80%-20% validation is chosen for the resulting pipeline.

## Using models for processing abstracts from PubMed

*key_sentence_predictor.py* contains the class *KeySentencePredictor* which can be used to process a list X of dictionaries containing keys "abstract" and "pmid". 
(This class is used by *key_sentence_predictor.py* and *semisuper/tests/key_sentence_predictor_test.py*)

Calling *transform(X)* or *predict(X)* return a dictionary containing a list of key sentences for each pmid, of the form: 

> {<pmid_i> : [ <start_i0, end_i0, score_i0>, <start_i1, end_i1, score_i1>, ... ], <pmid_i+1> : ... }

If *semisuper/pickles/semi_pipeline.pickle* or *semisuper/silver_standard.tsv* does not exist, a new pipeline and silver standard are generated.
