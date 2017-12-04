import time
import sys

import numpy as np
from scipy.sparse import vstack

from sklearn.model_selection import train_test_split
from semisuper import loaders, pu_two_step, pu_biased_svm, basic_pipeline, cleanup_sources
from semisuper.helpers import num_rows, densify, eval_model, run_fun
from semisuper.basic_pipeline import identitySelector, percentile_selector
from functools import partial
import multiprocessing as multi

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

def prepare_corpus(ratio=0.5):
    # remove worst percentage
    # print("\nRemoving CIViC-like sentences from HoC[neg]\n")
    # hocneg_ = cleanup_sources.remove_least_similar_percent(noisy=hocneg, guide=civic, ratio=ratio, percentile=15)
    # print("\nRemoving HoC[neg]-like sentences from HoC[pos]\n")
    # hocpos_ = cleanup_sources.remove_least_similar_percent(noisy=hocpos, guide=hocneg_, ratio=ratio, percentile=10)
    # print("\nRemoving CIViC-unlike sentences from HoC[pos]\n")
    # hocpos_ = cleanup_sources.remove_least_similar_percent(noisy=hocpos_, guide=civic, ratio=ratio, percentile=10,
    #                                                        inverse=True)

    # remove what is ambiguous according to PU training
    print("\nRemoving CIViC-like sentences from HoC[neg]\n")
    hocneg_ = cleanup_sources.remove_P_from_U(noisy=hocneg, guide=civic, ratio=ratio)

    print("\nRemoving HoC[neg]-like sentences from HoC[pos]\n")
    hocpos_ = cleanup_sources.remove_P_from_U(noisy=hocpos, guide=hocneg_, ratio=ratio)

    # print("\nRemoving CIViC-unlike sentences from HoC[pos]\n")
    # hocpos_ = cleanup_sources.remove_P_from_U(noisy=hocpos, guide=civic, ratio=ratio, inverse=True)

    hocpos_train, hocpos_test = train_test_split(hocpos_, test_size=0.2)
    civic_train, civic_test = train_test_split(civic, test_size=0.2)

    hocneg_train, X_test_neg = train_test_split(hocneg_, test_size=0.2)

    P_raw = np.concatenate((hocpos_train, civic_train))
    U_raw = np.concatenate((abstracts, hocneg_train))

    X_test_pos = np.concatenate((hocpos_test, civic_test))

    if ratio < 1.0:
        P_raw, _ = train_test_split(P_raw, train_size=ratio)
        U_raw, _ = train_test_split(U_raw, train_size=ratio)
        X_test_pos, _ = train_test_split(X_test_pos, train_size=ratio)
        X_test_neg, _ = train_test_split(X_test_neg, train_size=ratio)

    X_test_raw = np.concatenate((X_test_pos, X_test_neg))
    y_test = np.concatenate((np.ones(num_rows(X_test_pos)), np.zeros(num_rows(X_test_neg))))

    print("\nPU TRAINING", "(on", 100 * ratio, "% of available data)",
          "\tP: HOC POS + CIVIC", "(", num_rows(P_raw), ")",
          "\tN: HOC NEG + ABSTRACTS (", num_rows(U_raw), ")",
          "\tTEST SET (HOC POS + CIVIC + HOC NEG):", num_rows(X_test_raw)
          )

    # wordgram_range = (1, 4) # TODO change back to True, (1,4)
    # chargram_range = (2, 6) # TODO change back to True, (2,6)
    # min_df_word, min_df_char = [20, 30]  # TODO change back to default(20,20)
    # rules, lemmatize = [True, True]
    #
    # def print_params():
    #     print("word n-gram range:", wordgram_range,
    #           "\nchar n-gram range:", chargram_range,
    #           "\nrule-based preprocessing:", rules, "\tlemmatization:", lemmatize)
    #     return
    #
    # print_params()
    #
    # print("Fitting vectorizer")
    # vec = basic_pipeline.vectorizer(wordgrams=wordgram_range,
    #                                 chargrams=chargram_range, rules=rules, lemmatize=lemmatize,
    #                                 min_df_word=min_df_word, min_df_char=min_df_char)
    vec = basic_pipeline.vectorizer()
    vec.fit(np.concatenate((P_raw, U_raw)))

    P = vec.transform(P_raw)
    U = vec.transform(U_raw)

    print("Features before selection:", np.shape(P)[1])

    # sel = identitySelector()
    sel = percentile_selector(percentile=20)
    # sel = basic_pipeline.factorization('LatentDirichletAllocation')

    sel.fit(vstack((P, U)),
            (np.concatenate((np.ones(num_rows(P)), np.zeros(num_rows(U))))))
    P = densify(sel.transform(P))
    U = densify(sel.transform(U))
    X_test = densify(sel.transform(vec.transform(X_test_raw)))

    print("Features after selection:", np.shape(P)[1])

    return P, U, X_test, y_test, vec, sel


# ------------------
# execute
# ------------------

P, U, X_test, y_test, vectorizer, selector = prepare_corpus(1.0)
test_all(P, U, X_test, y_test, sample_sentences=True)
