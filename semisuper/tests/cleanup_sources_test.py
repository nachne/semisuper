import random
import time

import numpy as np

from sklearn.model_selection import train_test_split
from semisuper import loaders, pu_two_step, pu_biased_svm, ss_techniques
from semisuper.helpers import num_rows, densify, eval_model, run_fun, pu_score, select_PN_below_score
from semisuper.cleanup_corpora import *
from semisuper.transformers import IdentitySelector
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

print("\nHOCNEG \ CIVIC\n")
_ = remove_most_similar_percent(civic, hocneg, ratio=1.0, percentile=20)

print("\nCIVIC * HOCPOS\n")
civic_ = remove_most_similar_percent(hocpos, civic, ratio=1.0, inverse=True)

print("\nHOCPOS * CIVIC\n")
hocpos_ = remove_most_similar_percent(civic, hocpos, ratio=1.0, inverse=True)

print("\nCIVIC * HOCPOS'\n")
_ = remove_most_similar_percent(hocpos_, civic, ratio=1.0, inverse=True)

print("\nHOCPOS * CIVIC'\n")
_ = remove_most_similar_percent(civic_, hocpos, ratio=1.0, inverse=True)

print("\nHOCNEG \ HOCPOS'\n")
_ = remove_most_similar_percent(hocpos_, hocneg, ratio=1.0, percentile=20)

# sys.exit()

# PU hocneg vs civic: ok, 6%

print("\nHOCNEG \ CIVIC\n")
hocneg_ = remove_P_from_U(U=hocneg, P=civic, ratio=0.2)
# print_sentences(model, "best PU method (remove civic from hocneg)")

print("\nHOCNEG \ HOCPOS\n")
_ = remove_P_from_U(U=hocneg, P=hocpos, ratio=0.2)
# print_sentences(model, "best PU method (remove hocpos from hocneg)")

print("\nHOCNEG \ CIVIC \ HOCPOS\n")
_ = remove_P_from_U(U=hocneg_, P=hocpos, ratio=0.2)
# print_sentences(model, "best PU method (remove hocpos from hocneg_)")

print("\nHOCPOS \ (HOCNEG \ CIVIC)\n")
hocpos_ = remove_P_from_U(U=hocpos, P=hocneg_, ratio=0.2)
# print_sentences(model, "best PU method (remove hocneg_ from hocpos)")

print("\nHOCPOS \ HOCNEG\n")
_ = remove_P_from_U(U=hocpos, P=hocneg, ratio=0.2)
# print_sentences(model, "best PU method (remove hocneg from hocpos)")

print("\nHOCPOS * CIVIC\n")
_ = remove_P_from_U(U=hocpos, P=civic, ratio=0.2, inverse=True)
# print_sentences(model, "best PU method (keep civic in hocpos)")

print("\nCIVIC * HOCPOS\n")
_ = remove_P_from_U(U=civic, P=hocpos, ratio=0.2, inverse=True)
# print_sentences(model, "best PU method (keep hocpos in civic)")

print("\nCIVIC * (HOCPOS \ (HOCNEG \ CIVIC))\n")
_ = remove_P_from_U(U=civic, P=hocpos_, ratio=0.2, inverse=True)
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
