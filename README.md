# Semisuper
Sentence classification for oncological abstracts **(requires pre-trained models)**

# Prerequisites

## Installation via setup.py

Installation requires a Unix system with *python3*, *pip3*, and Python3 dependencies *setuptools* and *nltk* installed.
(ideally *scipy* and *scikit-learn* are installed already, as their dependencies can be problematic. Tested on macOS and Ubuntu)

> python3 setup.py

installs additional Python dependencies with pip (if necessary and possible), and downloads nltk data packages.

## Manual installation

(working directory = from the project root containing this file)

* mkdir semisuper/pickles semisuper/silver_standard

* Install Python dependencies (can be installed using pip3; scipy may require additional math libraries):
    * numpy==1.13.3
    * scipy==0.19.1
    * scikit-learn==0.19.1
    * pandas==0.20.3
    * Unidecode==0.4.21
    * biopython==1.70
    * nltk==3.2.4

# Usage

## Demo 

> python3 demo.py [<max_abstracts>]

runs a simple demo.
A random sample (optional parameter *max_abstracts*, default 400) of preloaded articles is processed by a pretrained classifier. Results are ouput as *demo.html*, a static HTML file where these articles' key sentences are highlighted, with opacity according to prediction confidence.

If *semisuper/pickles/sent_test_abstracts.pickle* does not exist, new articles for the query "cancer" are fetched from PubMed.
It is a good idea to use large max_abstracts (e.g. 1000, 200K, but not just 20) once, so that there is a pool of downloaded abstracts for subsequent runs to randomly draw from. Otherwise, small samples of the latest PubMed cancer articles may have obscure topics.

*/semisuper/pickles/semi_pipeline.pickle* and */semisuper/silver_standard/silver_standard.tsv* are required as this branch does not contain training modules.

## Using models for processing abstracts from PubMed (in other programs)

*key_sentence_predictor.py* contains the class *KeySentencePredictor* which can be used to process a list X of dictionaries containing keys "abstract" and "pmid". 
(This class is used by *demo.py* and *semisuper/tests/key_sentence_predictor_test.py*)

Calling *transform(X)* or *predict(X)* return a dictionary containing a list of key sentences for each pmid, of the form: 

> {<pmid_i> : [ <start_i0, end_i0, score_i0>, <start_i1, end_i1, score_i1>, ... ], <pmid_i+1> : ... }

If *semisuper/pickles/semi_pipeline.pickle* or *semisuper/silver_standard.tsv* does not exist, a new pipeline and silver standard are generated.
# semisuper_py2
