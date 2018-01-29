from __future__ import absolute_import, division, print_function

import time

import numpy as np

from semisuper import loaders, pu_two_step, pu_biased_svm, cleanup_corpora
from semisuper.helpers import num_rows, densify, eval_model

civic, abstracts = loaders.sentences_civic_abstracts()
hocpos, hocneg = loaders.sentences_HoC()
piboso_other = loaders.sentences_piboso_other()
piboso_outcome = loaders.sentences_piboso_outcome()

print("CIViC sentences:", len(civic))
print("Abstract sentences:", len(abstracts))
print("HoC positive sentences:", len(hocpos))
print("HoC negative sentences:", len(hocneg))
print("PIBOSO outcome sentences:", len(piboso_outcome))
print("PIBOSO other sentences:", len(piboso_other))


# ------------------
# model testers
# ------------------

def test_all(P, U, X_test=None, y_test=None, sample_sentences=False):
    # test_i_em(P, U, X_test, y_test, sample_sentences)
    test_s_em(P, U, X_test, y_test, sample_sentences)
    test_roc_svm(P, U, X_test, y_test, sample_sentences)
    test_cr_svm(P, U, X_test, y_test, sample_sentences)
    # test_roc_em(P, U, X_test, y_test, sample_sentences)
    test_spy_svm(P, U, X_test, y_test, sample_sentences)
    test_biased_svm(P, U, X_test, y_test, sample_sentences)
    test_biased_svm_grid(P, U, X_test, y_test, sample_sentences)
    return


def test_s_em(P, U, X_test=None, y_test=None, sample_sentences=False):
    print("\n\n"
          "---------\n"
          "S-EM TEST\n"
          "---------\n")

    start_time = time.time()

    model = pu_two_step.s_EM(P, U, spy_ratio=0.15, tolerance=0.15, noise_lvl=0.1)
    print("\nTraining S-EM took %s seconds\n" % (time.time() - start_time))

    eval_model(model, X_test, y_test)

    if sample_sentences:
        print_sentences(model, "S-EM")
    return


def test_i_em(P, U, X_test=None, y_test=None, sample_sentences=False):
    print("\n\n"
          "---------\n"
          "I-EM TEST\n"
          "---------\n")

    start_time = time.time()

    model = pu_two_step.i_EM(P, U, max_pos_ratio=0.5, max_imbalance=100.0, tolerance=0.15)
    print("\nTraining I-EM took %s seconds\n" % (time.time() - start_time))

    eval_model(model, X_test, y_test)

    if sample_sentences:
        print_sentences(model, "I-EM")
    return


def test_roc_svm(P, U, X_test=None, y_test=None, sample_sentences=False):
    print("\n\n"
          "------------\n"
          "ROC-SVM TEST\n"
          "------------\n")

    start_time = time.time()
    model = pu_two_step.roc_SVM(P, U, max_neg_ratio=0.1)
    print("\nTraining ROC-SVM took %s seconds\n" % (time.time() - start_time))

    eval_model(model, X_test, y_test)

    if sample_sentences:
        print_sentences(model, "ROC-SVM")
    return


def test_cr_svm(P, U, X_test=None, y_test=None, sample_sentences=False):
    print("\n\n"
          "-----------\n"
          "CR-SVM TEST\n"
          "-----------\n")

    start_time = time.time()
    model = pu_two_step.cr_SVM(P, U, max_neg_ratio=0.1, noise_lvl=0.5)
    print("\nTraining CR-SVM took %s seconds\n" % (time.time() - start_time))

    eval_model(model, X_test, y_test)

    if sample_sentences:
        print_sentences(model, "CR-SVM")
    return


def test_spy_svm(P, U, X_test=None, y_test=None, sample_sentences=False):
    print("\n\n"
          "------------\n"
          "SPY-SVM TEST\n"
          "------------\n")

    start_time = time.time()
    model = pu_two_step.spy_SVM(P, U, spy_ratio=0.15, max_neg_ratio=0.1, tolerance=0.15, noise_lvl=0.2)
    print("\nTraining SPY-SVM took %s seconds\n" % (time.time() - start_time))

    eval_model(model, X_test, y_test)

    if sample_sentences:
        print_sentences(model, "SPY-SVM")
    return


def test_roc_em(P, U, X_test=None, y_test=None, sample_sentences=False):
    print("\n\n"
          "-----------\n"
          "ROC-EM TEST\n"
          "-----------\n")

    start_time = time.time()
    model = pu_two_step.roc_EM(P, U, max_pos_ratio=0.5, tolerance=0.1, clf_selection=True)
    print("\nTraining ROC-EM took %s seconds\n" % (time.time() - start_time))

    eval_model(model, X_test, y_test)

    if sample_sentences:
        print_sentences(model, "ROC-EM")
    return


def test_biased_svm(P, U, X_test=None, y_test=None, sample_sentences=False):
    print("\n\n"
          "---------------\n"
          "BIASED-SVM TEST (C+, C-, C)\n"
          "---------------\n")

    start_time = time.time()

    model = pu_biased_svm.biased_SVM_weight_selection(P, U,
                                                      Cs=[10 ** x for x in range(1, 5, 1)],
                                                      Cs_neg=[1],
                                                      Cs_pos_factors=range(1, 1100, 200))

    print("\nTraining Biased-SVM took %s seconds\n" % (time.time() - start_time))

    eval_model(model, X_test, y_test)

    if sample_sentences:
        print_sentences(model, "BIASED-SVM")
    return


def test_biased_svm_grid(P, U, X_test=None, y_test=None, sample_sentences=False):
    print("\n\n"
          "---------------\n"
          "BAGGING-SVM TEST (GRID SEARCH FOR C AS DESCRIBED BY MORDELET)\n"
          "---------------\n")

    start_time = time.time()

    model = pu_biased_svm.biased_SVM_grid_search(P, U)

    print("\nTraining Biased-SVM took %s seconds\n" % (time.time() - start_time))

    eval_model(model, X_test, y_test)

    if sample_sentences:
        print_sentences(model, "BAGGING-SVM")
    return


# ------------------
# helper to print amount of positive docs, top and bottom sentences per corpus
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
# prepare corpus, vectors, vectorizer, selector
# ------------------


# ------------------
# execute
# ------------------


if __name__ == "__main__":
    P_, U_, X_test, y_test, vectorizer, selector = cleanup_corpora.vectorized_clean_pu(0.1)
    test_all(P_, U_, X_test, y_test, sample_sentences=True)
