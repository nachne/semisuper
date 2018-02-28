# coding=utf-8
from __future__ import absolute_import, division, print_function

# import cPickle as pickle
# import os
# import time
# import multiprocessing
# from tqdm import tqdm
# import random
# from itertools import cycle
#
# from semisuper import loaders, helpers

# READ MEDLINE
import pubmed_parser as pp

path = './semisuper/pubmedgz/pubmed18n0001.xml.gz'
# path = '/home/docClass/files/pubmed/pubmed18n0011.xml.gz'
articles = pp.parse_medline_xml(path)
print("read medline file")

# PREDICTOR
print("importing predictor")
import key_sentence_predictor

print("imported predictor")
predictor = key_sentence_predictor.KeySentencePredictor(batch_size=400)
print("created keysentencepredictor object")

predicted = predictor.transform(articles)
print(predicted, type(predicted))

scores = [x[2] for x in [v for k, v in predicted.items() if len(v)] if len(x) > 2]

print(scores)
