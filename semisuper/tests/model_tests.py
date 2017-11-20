import random
import time

import numpy as np
from semisuper import loaders, pu_two_step, pu_biased_svm, basic_pipeline
from semisuper.helpers import num_rows, unsparsify

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

# P_raw = civic
# U_raw = abstracts

# print("\n\nTRAINING ON CIVIC AND ABSTRACTS\n\n")

show_sentences = False

P_raw = random.sample(hocpos + civic, 400)
U_raw = random.sample(hocneg + abstracts, 400)

print("\nTRAINING ON HOC CORPUS"
      , ", CIVIC"
      , ", AND ABSTRACTS"
      )

words, wordgram_range = [True, (1, 1)]  # TODO change back to True, (1,3)
chars, chargram_range = [True, (1, 1)]  # TODO change back to True, (3,6)
rules, lemmatize = [True, True]

print("Fitting vectorizer")
vectorizer = basic_pipeline.feature_vectorizer(words=words, wordgram_range=wordgram_range,
                                               chars=chars, chargram_range=chargram_range,
                                               rules=rules, lemmatize=lemmatize)
vectorizer.fit(np.concatenate((P_raw, U_raw)))
P = unsparsify(vectorizer.transform(P_raw))
U = unsparsify(vectorizer.transform(U_raw))

print("P:", num_rows(P), "\tU:", num_rows(U))


# ------------------
# model testers
# ------------------

def test_all(P, U):
    test_biased_svm(P, U)
    test_i_em(P, U)
    test_s_em(P, U)
    test_roc_svm(P, U)
    test_cr_svm(P, U)
    test_roc_em(P, U)
    test_spy_svm(P, U)
    return


def test_s_em(P, U):
    print("\n\n"
          "---------\n"
          "S-EM TEST\n"
          "---------\n")

    start_time = time.time()

    model = pu_two_step.s_EM(P, U, spy_ratio=0.15, tolerance=0.15, noise_lvl=0.1)
    print("\nTraining S-EM took %s seconds\n" % (time.time() - start_time))

    print_sentences(model, "S-EM")
    return


def test_i_em(P, U):
    print("\n\n"
          "---------\n"
          "I-EM TEST\n"
          "---------\n")

    start_time = time.time()

    model = pu_two_step.i_EM(P, U, max_pos_ratio=0.5, max_imbalance=100.0, tolerance=0.15)
    print("\nTraining I-EM took %s seconds\n" % (time.time() - start_time))

    print_sentences(model, "I-EM")
    return


def test_roc_svm(P, U):
    print("\n\n"
          "------------\n"
          "ROC-SVM TEST\n"
          "------------\n")

    start_time = time.time()
    model = pu_two_step.roc_SVM(P, U, max_neg_ratio=0.1)
    print("\nTraining ROC-SVM took %s seconds\n" % (time.time() - start_time))

    print_sentences(model, "ROC-SVM")
    return


def test_cr_svm(P, U):
    print("\n\n"
          "-----------\n"
          "CR-SVM TEST\n"
          "-----------\n")

    start_time = time.time()
    model = pu_two_step.cr_SVM(P, U, max_neg_ratio=0.1, noise_lvl=0.5)
    print("\nTraining CR-SVM took %s seconds\n" % (time.time() - start_time))

    print_sentences(model, "CR-SVM")
    return


def test_spy_svm(P, U):
    print("\n\n"
          "------------\n"
          "SPY-SVM TEST\n"
          "------------\n")

    start_time = time.time()
    model = pu_two_step.spy_SVM(P, U, spy_ratio=0.15, max_neg_ratio=0.1, tolerance=0.15, noise_lvl=0.2)
    print("\nTraining SPY-SVM took %s seconds\n" % (time.time() - start_time))

    print_sentences(model, "SPY-SVM")
    return


def test_roc_em(P, U):
    print("\n\n"
          "-----------\n"
          "ROC-EM TEST\n"
          "-----------\n")

    start_time = time.time()
    model = pu_two_step.roc_EM(P, U, max_pos_ratio=0.5, tolerance=0.1, clf_selection=True)
    print("\nTraining ROC-EM took %s seconds\n" % (time.time() - start_time))

    print_sentences(model, "ROC-EM")
    return


def test_biased_svm(P, U):
    print("\n\n"
          "---------------\n"
          "BIASED-SVM TEST\n"
          "---------------\n")

    start_time = time.time()
    model = pu_biased_svm.biased_SVM_weight_selection(P, U,
                                                      Cs=[10 ** x for x in range(1, 5, 1)],
                                                      Cs_neg=[1],
                                                      Cs_pos_factors=range(1, 1100, 200))
    print("\nTraining Biased-SVM took %s seconds\n" % (time.time() - start_time))

    print_sentences(model, "BIASED-SVM")
    return


# ------------------
# helper to print amount of positive docs, top and bottom sentences per corpus
# ------------------

def print_sentences(model, modelname=""):
    if not show_sentences:
        return

    print("\n\n"
          "----------------\n"
          "{} SENTENCES\n"
          "----------------\n".format(modelname))

    def sort_model(sentences):
        sent_features = unsparsify(vectorizer.transform(sentences))

        return sorted(zip(model.predict_proba(sent_features),
                          sentences),
                      key=lambda x: x[0][1],
                      reverse=True)

    def top_bot_12_model(predictions, name):
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

def print_params():
    print("words:", words, "\tword n-gram range:", wordgram_range,
          "\nchars:", chars, "\tchar n-gram range:", chargram_range,
          "\nrule-based preprocessing:", rules, "\tlemmatization:", lemmatize)
    return


# ------------------
# execute
# ------------------

print_params()
test_all(P, U)
