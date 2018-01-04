import time

import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from semisuper.cleanup_corpora import vectorized_clean_pnu
from semisuper import loaders, ss_techniques
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

def test_all(P, N, U, X_test=None, y_test=None, sample_sentences=False, outpath=None):
    test_iterative_linearSVM(P, N, U, X_test, y_test, sample_sentences)

    test_neg_self_training(P, N, U, X_test, y_test, sample_sentences)

    test_self_training_linsvc(P, N, U, X_test, y_test, sample_sentences)

    test_self_training(P, N, U, X_test, y_test, sample_sentences)

    # test_knn(P, N, U, X_test, y_test, sample_sentences)
    # test_em(P, N, U, X_test, y_test, sample_sentences)

    # supervised thingies
    test_supervised(P, N, U, X_test, y_test, sample_sentences)
    # test_svc(P, N, U, X_test, y_test, sample_sentences)
    # test_linearSVM(P, N, U, X_test, y_test, sample_sentences)

    # poly, rbf, sigmoid go wrong
    # test_iterative_SVC(P, N, U, X_test, y_test, sample_sentences, kernel="poly")
    # test_iterative_SVC(P, N, U, X_test, y_test, sample_sentences, kernel="linear")
    # test_iterative_SVC(P, N, U, X_test, y_test, sample_sentences, kernel="rbf")
    # test_iterative_SVC(P, N, U, X_test, y_test, sample_sentences, kernel="sigmoid")

    # test_knn(P, N, U, X_test, y_test, sample_sentences)
    # test_em(P, N, U, X_test, y_test, sample_sentences)
    # test_label_propagation(P, N, U, X_test, y_test)

    return


def test_supervised(P, N, U, X_test=None, y_test=None, sample_sentences=False):
    print("\n\n"
          "---------\n"
          "SUPERVISED TECHNIQUES TEST\n"
          "---------\n")

    clfs = {
        'sgd': ss_techniques.sgd,
        # 'logreg' : ss_techniques.logreg, # grid search too slow
        # 'mlp': ss_techniques.mlp,
        # 'dectree': ss_techniques.dectree,
        # 'mnb'         : ss_techniques.mnb,
        # 'randomforest': ss_techniques.randomforest
        # 'linsvc': ss_techniques.grid_search_linearSVM
    }

    for name in clfs:
        start_time = time.time()
        print("\nSupervised training with", name)
        model = clfs[name](P, N, U=None)
        print("took", time.time() - start_time, "seconds")
        eval_model(model, X_test, y_test)
        if sample_sentences:
            print_sentences(model, name)
    return


def test_self_training(P, N, U, X_test=None, y_test=None, sample_sentences=False, clf=None, confidence=0.8):
    print("\n\n"
          "---------\n"
          "SELF-TRAINING TEST\n"
          "---------\n")

    start_time = time.time()

    model = ss_techniques.self_training(P, N, U, clf=clf, confidence=confidence)

    print("\nIterating Self-Training with", (clf or "Logistic Regression"),
          "confidence =", confidence, "took %s\n" % (time.time() - start_time), "seconds")

    eval_model(model, X_test, y_test)

    if sample_sentences:
        print_sentences(model, "Self-Training {} {}".format((clf or "Logistic Regression"), confidence))
    return


def test_self_training_linsvc(P, N, U, X_test=None, y_test=None, sample_sentences=False, clf=None, confidence=0.5):
    print("\n\n"
          "---------\n"
          "SELF-TRAINING SVC TEST\n"
          "---------\n")

    start_time = time.time()

    model = ss_techniques.self_training_lin_svc(P, N, U, confidence=confidence, clf=clf)

    print("\nIterating Self-Training with", (clf or "Linear SVC"),
          "confidence =", confidence, "took %s\n" % (time.time() - start_time), "seconds")

    eval_model(model, X_test, y_test)

    if sample_sentences:
        print_sentences(model, "Self-Training {} {}".format((clf or "LinearSVC"), confidence))
    return


def test_neg_self_training(P, N, U, X_test=None, y_test=None, sample_sentences=False, clf=None):
    print("\n\n"
          "---------\n"
          "NEG SELF-TRAINING TEST\n"
          "---------\n")

    start_time = time.time()
    model = ss_techniques.neg_self_training(P, N, U, SGDClassifier(loss='modified_huber'))
    print("\nIteratively expanding negative set with SGDClassifier",
          "took %s\n" % (time.time() - start_time), "seconds")
    eval_model(model, X_test, y_test)
    if sample_sentences:
        print_sentences(model, "Negative Self-Training SGD")

    # start_time = time.time()
    # model = ss_techniques.neg_self_training(P, N, U, clf=clf)
    # print("\nIteratively expanding negative set with", (clf or "Logistic Regression"),
    #       "took %s\n" % (time.time() - start_time), "seconds")
    # eval_model(model, X_test, y_test)
    # if sample_sentences:
    #     print_sentences(model, "Negative Self-Training {}".format((clf or "Logistic Regression")))

    return


def test_iterative_SVC(P, N, U, X_test=None, y_test=None, sample_sentences=False, kernel='rbf'):
    print("\n\n"
          "---------\n"
          "ITERATIVE SVM TEST\n"
          "---------\n")

    start_time = time.time()

    model = ss_techniques.iterate_SVC(P, N, U, kernel=kernel)

    print("\nIterating SVC with", kernel, "kernel took %s seconds\n" % (time.time() - start_time))

    eval_model(model, X_test, y_test)

    if sample_sentences:
        print_sentences(model, "iterativeSVM")
    return


def test_iterative_linearSVM(P, N, U, X_test=None, y_test=None, sample_sentences=False):
    print("\n\n"
          "---------\n"
          "ITERATIVE LINEAR SVM TEST\n"
          "---------\n")

    start_time = time.time()

    model = ss_techniques.iterate_linearSVC(P, N, U, 1.0)

    print("\nIterating SVM took %s seconds\n" % (time.time() - start_time))

    eval_model(model, X_test, y_test)

    if sample_sentences:
        print_sentences(model, "iterativeSVM")
    return


def test_linearSVM(P, N, U, X_test=None, y_test=None, sample_sentences=False):
    print("\n\n"
          "---------\n"
          "LINEAR SVM TEST\n"
          "---------\n")

    start_time = time.time()

    model = ss_techniques.grid_search_linearSVM(P, N, U)

    print("\nTraining grid-search linear SVC took %s seconds\n" % (time.time() - start_time))

    eval_model(model, X_test, y_test)

    if sample_sentences:
        print_sentences(model, "linearSVC")
    return


def test_svc(P, N, U, X_test=None, y_test=None, sample_sentences=False):
    print("\n\n"
          "---------\n"
          "SVC TEST\n"
          "---------\n")

    start_time = time.time()

    model = ss_techniques.grid_search_SVC(P, N, U)

    print("\nTraining grid-search SVC took %s seconds\n" % (time.time() - start_time))

    eval_model(model, X_test, y_test)

    if sample_sentences:
        print_sentences(model, "SVC")
    return


def test_knn(P, N, U, X_test=None, y_test=None, sample_sentences=False):
    print("\n\n"
          "---------\n"
          "ITERATIVE KNN TEST\n"
          "---------\n")

    start_time = time.time()

    model = ss_techniques.iterate_knn(P, N, U)

    print("\nTraining kNN self-training took %s seconds\n" % (time.time() - start_time))

    eval_model(model, X_test, y_test)

    if sample_sentences:
        print_sentences(model, "kNN")
    return


def test_label_propagation(P, N, U, X_test=None, y_test=None, sample_sentences=False):
    print("\n\n"
          "---------\n"
          "LABEL PROPAGATION TEST\n"
          "---------\n")

    start_time = time.time()

    model = ss_techniques.propagate_labels(P, N, U)

    print("\nTraining Label Propagation took %s seconds\n" % (time.time() - start_time))

    eval_model(model, X_test, y_test)

    if sample_sentences:
        print_sentences(model, "Label Propagation")
    return


# tends to separate civ+abs from hocpos+hocneg, rather than the other way round
def test_em(P, N, U, X_test=None, y_test=None, sample_sentences=False):
    print("\n\n"
          "---------\n"
          "EM TEST\n"
          "---------\n")

    start_time = time.time()

    model = ss_techniques.EM(P, N, U)

    print("\nTraining EM took %s seconds\n" % (time.time() - start_time))

    eval_model(model, X_test, y_test)

    if sample_sentences:
        print_sentences(model, "EM")
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
#  execute
# ------------------


P, N, U, vectorizer, selector = vectorized_clean_pnu(ratio=1.0)
P, X_test_pos = train_test_split(P, test_size=0.2)
N, X_test_neg = train_test_split(N, test_size=0.2)
test_all(P, N, U,
         X_test=np.concatenate((X_test_pos, X_test_neg)),
         y_test=np.concatenate((np.ones(num_rows(X_test_pos)), np.zeros(num_rows(X_test_neg)))),
         sample_sentences=True)
