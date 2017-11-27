import random
import time

import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.model_selection import train_test_split
from semisuper import loaders, basic_pipeline, ss_techniques
from semisuper.helpers import num_rows, unsparsify, eval_model
from semisuper.basic_pipeline import identitySelector, percentile_selector, factorization

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

def test_all(P, N, U, X_test=None, y_test=None, sample_sentences=False):

    test_neg_self_training(P, N, U, X_test, y_test, sample_sentences)

    # test_self_training(P, N, U, X_test, y_test, sample_sentences)

    # supervised thingies
    test_supervised(P, N, U, X_test, y_test, sample_sentences)
    # test_svc(P, N, U, X_test, y_test, sample_sentences)
    # test_linearSVM(P, N, U, X_test, y_test, sample_sentences)

    test_iterative_linearSVM(P, N, U, X_test, y_test, sample_sentences)

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
        'sgd'    : ss_techniques.sgd,
        'logreg' : ss_techniques.logreg,
        'mlp'    : ss_techniques.mlp,
        # 'dectree': ss_techniques.dectree,
        # 'mnb'         : ss_techniques.mnb,
        # 'randomforest': ss_techniques.randomforest
    }

    for name in clfs:
        start_time = time.time()
        print("\nSupervised training with", name)
        model = clfs[name](P, N, U=None)
        print("took", time.time() - start_time, "secs")
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

    model = ss_techniques.self_training(P, N, U, confidence=confidence, clf=clf)

    print("\nIterating Self-Training with", (clf or "Logistic Regression"),
          "confidence =", confidence, "took %s\n" % (time.time() - start_time))

    eval_model(model, X_test, y_test)

    if sample_sentences:
        print_sentences(model, "Self-Training {} {}".format((clf or "Logistic Regression"), confidence))
    return

def test_neg_self_training(P, N, U, X_test=None, y_test=None, sample_sentences=False, clf=None):
    print("\n\n"
          "---------\n"
          "NEG SELF-TRAINING TEST\n"
          "---------\n")

    start_time = time.time()
    model = ss_techniques.neg_self_training(P, N, U, clf=clf)
    print("\nIteratively expanding negative set with", (clf or "Logistic Regression"),
          "took %s\n" % (time.time() - start_time))
    eval_model(model, X_test, y_test)
    if sample_sentences:
        print_sentences(model, "Negative Self-Training {}".format((clf or "Logistic Regression")))

    start_time = time.time()
    model = ss_techniques.neg_self_training_sgd(P, N, U)
    print("\nIteratively expanding negative set with SGDClassifier",
          "took %s\n" % (time.time() - start_time))
    eval_model(model, X_test, y_test)
    if sample_sentences:
        print_sentences(model, "Negative Self-Training SGD")

    start_time = time.time()
    model = ss_techniques.neg_self_training_mlp(P, N, U,)
    print("\nIteratively expanding negative set with MLPClassifier",
          "took %s\n" % (time.time() - start_time))
    eval_model(model, X_test, y_test)
    if sample_sentences:
        print_sentences(model, "Negative Self-Training MLP")

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

    model = ss_techniques.iterate_linearSVC(P, N, U)

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
        sent_features = unsparsify(selector.transform(vectorizer.transform(sentences)))

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
    hocpos_train, X_test_pos = train_test_split(hocpos, test_size=0.2)
    hocneg_train, X_test_neg = train_test_split(hocneg, test_size=0.2)
    civic_train, civic_test = train_test_split(civic, test_size=0.2)

    P_raw = hocpos_train + civic_train
    U_raw = abstracts
    N_raw = hocneg_train

    if ratio < 1.0:
        P_raw = random.sample(P_raw, int(ratio * num_rows(P_raw)))
        N_raw = random.sample(N_raw, int(ratio * num_rows(N_raw)))
        U_raw = random.sample(U_raw, int(ratio * num_rows(U_raw)))
        X_test_pos = random.sample(X_test_pos, int(ratio * num_rows(X_test_pos)))
        X_test_neg = random.sample(X_test_neg, int(ratio * num_rows(X_test_neg)))

    X_test_raw = X_test_pos + X_test_neg
    y_test = np.concatenate((np.ones(num_rows(X_test_pos)), np.zeros(num_rows(X_test_neg))))

    print("\nSEMI-SUPERVISED TRAINING", "(on", 100 * ratio, "% of available data)",
          "\tP: HOC POS"
          , "+ CIVIC"
          , "(", num_rows(P_raw), ")"
          , "\tN: HOC NEG"
          , "(", num_rows(N_raw), ")"
          , "\tU: ABSTRACTS"
          , "(", num_rows(U_raw), ")"
          , "TEST SET (HOC POS + HOC NEG):", num_rows(X_test_raw)
          )

    words, wordgram_range = [True, (1, 4)]  # TODO change back to True, (1,3)
    chars, chargram_range = [True, (2, 6)]  # TODO change back to True, (3,6)
    rules, lemmatize = [True, True]

    def print_params():
        print("words:", words, "\tword n-gram range:", wordgram_range,
              "\nchars:", chars, "\tchar n-gram range:", chargram_range,
              "\nrule-based preprocessing:", rules, "\tlemmatization:", lemmatize)
        return

    print_params()

    print("Fitting vectorizer")
    vectorizer = basic_pipeline.vectorizer(words=words, wordgram_range=wordgram_range, chars=chars,
                                           chargram_range=chargram_range, rules=rules, lemmatize=lemmatize)
    vectorizer.fit(np.concatenate((P_raw, N_raw, U_raw)))

    P = (vectorizer.transform(P_raw))
    N = (vectorizer.transform(N_raw))
    U = (vectorizer.transform(U_raw))
    X_test = vectorizer.transform(X_test_raw)

    print("Features before selection:", np.shape(P)[1])

    selector = identitySelector()  # TODO FIXME chi2 does not help, PCA too slow
    # selector = basic_pipeline.selector()
    selector.fit(vstack((P, N, U)),
                 (np.concatenate((np.ones(num_rows(P)), -np.ones(num_rows(N)), np.zeros(num_rows(U))))))
    P = unsparsify(selector.transform(P))
    N = unsparsify(selector.transform(N))
    U = unsparsify(selector.transform(U))
    X_test = unsparsify(selector.transform(X_test))

    # print("Features after selection:", np.shape(P)[1])

    return P, N, U, X_test, y_test, vectorizer, selector  # ------------------


# execute
# ------------------

P, N, U, X_test, y_test, vectorizer, selector = prepare_corpus(1.0)
test_all(P, N, U, X_test, y_test, sample_sentences=True)
