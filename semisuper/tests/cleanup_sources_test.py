import random
import time

import numpy as np

from sklearn.model_selection import train_test_split
from semisuper import loaders, pu_two_step, pu_biased_svm, basic_pipeline, ss_techniques
from semisuper.helpers import num_rows, densify, eval_model, run_fun, pu_score, select_PN_below_score
from semisuper.cleanup_sources import *
from basic_pipeline import identitySelector, show_most_informative_features
from functools import partial
import multiprocessing as multi
import sys

civic, abstracts = loaders.sentences_civic_abstracts()
hocpos, hocneg = loaders.sentences_HoC()
piboso_other = loaders.sentences_piboso_other()
piboso_outcome = loaders.sentences_piboso_outcome()


# ------------------
# display
# ------------------

def print_sentences(model, modelname=""):
    print("\n\n"
          "----------------\n"
          "{} SENTENCES\n"
          "----------------\n".format(modelname))

    def sort_model(sentences):
        sent_features = densify(selector.transform(vectorizer.transform(sentences)))

        if hasattr(model, 'predict_proba'):
            return sorted(zip(model.predict_proba(sent_features),
                              sentences),
                          key=lambda x: x[0][1],
                          reverse=True)
        else:
            return sorted(zip(model.decision_function(sent_features),
                              sentences),
                          key=lambda x: x[0],
                          reverse=True)

    def top_bot_12_model(predictions, name):

        if np.isscalar(predictions[0][0]):
            pos = sum([1 for x in predictions if x[0] > 0])
        else:
            pos = sum([1 for x in predictions if x[0][1] > x[0][0]])
        num = num_rows(predictions)

        print()
        print(modelname, name, "prediction", pos, "/", num, "(", pos / num, ")")
        print()
        print(modelname, name, "top 12 (sentences with highest ranking)\n")
        [print(x) for x in (predictions[0:12])]
        print()
        print(modelname, name, "bottom 12 (sentences with lowest ranking)\n")
        [print(x) for x in (predictions[-12:])]

    civ_labelled = sort_model(civic)
    top_bot_12_model(civ_labelled, "civic")

    abs_labelled = sort_model(abstracts)
    top_bot_12_model(abs_labelled, "abstracts")

    hocpos_labelled = sort_model(hocpos)
    top_bot_12_model(hocpos_labelled, "HoC-pos")

    hocneg_labelled = sort_model(hocneg)
    top_bot_12_model(hocneg_labelled, "HoC-neg")

    out_labelled = sort_model(piboso_outcome)
    top_bot_12_model(out_labelled, "piboso-outcome")

    oth_labelled = sort_model(piboso_other)
    top_bot_12_model(oth_labelled, "piboso-other")

    return


# ------------------
# execute
# ------------------

print("\nHOCNEG MINUS CIVIC\n")
_ = remove_least_similar_percent(hocneg, civic, ratio=1.0, percentile=20)

print("\nCIVIC MINUS HOCPOS\n")
civic_ = remove_least_similar_percent(civic, hocpos, ratio=1.0, inverse=True)

print("\nHOCPOS MINUS CIVIC\n")
hocpos_ = remove_least_similar_percent(hocpos, civic, ratio=1.0, inverse=True)

print("\nCIVIC MINUS HOCPOS'\n")
_ = remove_least_similar_percent(civic, hocpos_, ratio=1.0, inverse=True)

print("\nHOCPOS MINUS CIVIC'\n")
_ = remove_least_similar_percent(hocpos, civic_, ratio=1.0, inverse=True)

sys.exit()

# PU hocneg vs civic: ok, 6%

hocneg_ = remove_most_similar(noisy_set=hocneg, guide_set=civic, ratio=0.2)
# print_sentences(model, "best PU method (remove civic from hocneg)")

_ = remove_most_similar(noisy_set=hocneg, guide_set=hocpos, ratio=0.2)
# print_sentences(model, "best PU method (remove hocpos from hocneg)")

_ = remove_most_similar(noisy_set=hocneg_, guide_set=hocpos, ratio=0.2)
# print_sentences(model, "best PU method (remove hocpos from hocneg_)")

hocpos_ = remove_most_similar(noisy_set=hocpos, guide_set=hocneg_, ratio=0.2)
# print_sentences(model, "best PU method (remove hocneg_ from hocpos)")

_ = remove_most_similar(noisy_set=hocpos, guide_set=hocneg, ratio=0.2)
# print_sentences(model, "best PU method (remove hocneg from hocpos)")

_ = remove_most_similar(noisy_set=hocpos, guide_set=civic, ratio=0.2, inverse=True)
# print_sentences(model, "best PU method (keep civic in hocpos)")

_ = remove_most_similar(noisy_set=civic, guide_set=hocpos, ratio=0.2, inverse=True)
# print_sentences(model, "best PU method (keep hocpos in civic)")

_ = remove_most_similar(noisy_set=civic, guide_set=hocpos_, ratio=0.2, inverse=True)
# print_sentences(model, "best PU method (keep hocpos_ in civic)")

sys.exit()

P, N, U, X_test, y_test, vectorizer, selector = prepare_corpus(ratio=0.5)

# SS civic vs hocneg -> hocpos yields 88% civic, 57% abstracts, 6% hocpos, 7% hocneg...
sys.exit()

it_lin_svc = ss_techniques.iterate_linearSVC(P, N, U)
eval_model(it_lin_svc, X_test, y_test)
print_sentences(it_lin_svc, "ITERATIVE LINEAR SVC")

em = ss_techniques.EM(P, N, U)
eval_model(it_lin_svc, X_test, y_test)
print_sentences(it_lin_svc, "EM")

knn = ss_techniques.iterate_knn(P, N, U)
eval_model(it_lin_svc, X_test, y_test)
print_sentences(it_lin_svc, "ITERATIVE KNN")
