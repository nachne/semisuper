import multiprocessing as multi
import random
from functools import partial

import numpy as np
from scipy.sparse import vstack
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC

from semisuper import loaders, pu_two_step, pu_cos_roc, pu_biased_svm, transformers
from semisuper.helpers import num_rows, densify, pu_score, select_PN_below_score

# ------------------
# globals
# ------------------

civic, abstracts = loaders.sentences_civic_abstracts()
hocpos, hocneg = loaders.sentences_HoC()
piboso_other = loaders.sentences_piboso_other()
piboso_outcome = loaders.sentences_piboso_outcome()

PARALLEL = True

genia_defaults = None # {"pos": False, "ner": False} # TODO safe default options

# ------------------
# select sentences
# ------------------

def remove_P_from_U(P, U, ratio=1.0, inverse=False, verbose=True):
    """Remove sentences from noisy_set that are similar to guide_set according to strictest PU estimator.

    if inverse is set to True, keep rather than discard them."""

    guide_, noisy_, vectorizer, selector = vectorize_preselection(P, U, ratio=ratio)

    model = best_pu(guide_, noisy_)

    y_noisy = model.predict(selector.transform(densify(vectorizer.transform(U))))

    if inverse:
        action = "Keeping"
        criterion = 1
    else:
        action = "Discarding"
        criterion = 0

    print(action, (100 * np.sum(y_noisy) / num_rows(y_noisy)), "% of noisy data (", np.sum(y_noisy), "sentences )",
          "as per result of PU learning")

    keeping = np.array([x for (x, y) in zip(U, y_noisy) if y == criterion], dtype=object)

    if verbose:
        discarding = [x for (x, y) in zip(U, y_noisy) if y != criterion]
        print("Keeping", random.sample(keeping.tolist(), 10))
        print("Discarding", random.sample(discarding, 10))

    return keeping


def remove_most_similar_percent(P, U, ratio=1.0, percentile=10, inverse=False, verbose=True):
    """Remove percentile of sentences from noisy_set that are similar to guide_set according to strictest PU estimator.

    if inverse is set to True, remove least rather than most similar."""

    guide_, noisy_, vectorizer, selector = vectorize_preselection(P, U, ratio=ratio)

    model = pu_cos_roc.ranking_cos_sim(guide_)

    if inverse:
        predicate = "least"
        y_pred = model.predict_proba(noisy_)
    else:
        predicate = "most"
        y_pred = -model.predict_proba(noisy_)

    print("Removing", percentile, "% of noisy data", predicate, "similar to guide set (cos-similarity)"
          , "(", (percentile * num_rows(U) / 100), "sentences )")

    U = np.array(U, dtype=object)
    U_minus_PN, PN = select_PN_below_score(y_pred, U, y_pred, noise_lvl=percentile / 100)

    if verbose:
        print("Keeping", train_test_split(U_minus_PN, train_size=10)[0])
        print("Discarding", train_test_split(PN, train_size=10)[0])

    return U_minus_PN


# ------------------
# model selection
# ------------------

def best_pu(P, U):
    P_train, P_test = train_test_split(P, test_size=0.2)
    U_train, U_test = train_test_split(U, test_size=0.2)

    models = [
        # {"name": "I-EM", "model": pu_two_step.i_EM},
        # {"name": "S-EM", "model": pu_two_step.s_EM},
        # {"name": "ROC-EM", "model": pu_two_step.roc_EM},
        {"name": "ROC-SVM", "model": pu_two_step.roc_SVM},
        {"name": "CR-SVM", "model": pu_two_step.cr_SVM},
        # {"name": "SPY-SVM", "model": pu_two_step.spy_SVM},
        # {"name": "ROCCHIO", "model": pu_two_step.rocchio},
        # {"name": "BIASED-SVM", "model": pu_biased_svm.biased_SVM_weight_selection},
    ]

    if PARALLEL:
        with multi.Pool(multi.cpu_count()) as p:
            stats = list(p.map(partial(model_pu_score_record, P_train, U_train, P_test, U_test), models))
    else:
        stats = list(map(partial(model_pu_score_record, P_train, U_train, P_test, U_test), models))

    for s in stats:
        print(s["name"], "\tPU-score:", s["pu_score"])

    best_model = max(stats, key=lambda x: x["pu_score"])
    print("Best PU model:", best_model["name"], "\tPU-score:", best_model["pu_score"])
    print("Retraining best PU model on all of P and U")

    return best_model["model"](P, U)


def model_pu_score_record(P_train, U_train, P_test, U_test, m):
    model = m['model'](P_train, U_train)
    name = m['name']

    y_pred = model.predict(np.concatenate((P_test, U_test)))
    y_P = y_pred[:num_rows(P_test)]
    y_U = y_pred[num_rows(P_test):]

    score = pu_score(y_P, y_U)

    return {'name': name, 'model': m['model'], 'pu_score': score}


# ------------------
# preprocess
# ------------------


def vectorize_preselection(P, U, ratio=1.0):
    """generate and select features for ratio of sentence sets"""

    print("Preprocessing corpora for PU learning")

    if ratio < 1.0:
        print("Training on", 100 * ratio, "% of data")
        P, _ = train_test_split(P, train_size=ratio)
        U, _ = train_test_split(U, train_size=ratio)

    vec = transformers.vectorizer(genia_opts=genia_defaults)
    vec.fit(np.concatenate((P, U)))

    P_ = vec.transform(P)
    U_ = vec.transform(U)

    print("Features before selection:", np.shape(U_)[1])

    sel = transformers.percentile_selector()
    # sel = basic_pipeline.factorization('TruncatedSVD', 1000)
    sel.fit(vstack((P_, U_)),
            np.concatenate((np.ones(num_rows(P_)), -np.ones(num_rows(U_)))))

    P_ = densify(sel.transform(P_))
    U_ = densify(sel.transform(U_))

    return P_, U_, vec, sel


def clean_corpus_pnu(mode=None, percentiles=(10, 25, 10), ratio=1.0):
    """clean up HoC corpus using PU learning. Modes: "strict", "percentile", default

    default: remove CIViC-like from HoC[neg], HoC[neg]-like from CIViC
    strict: remove CIViC-like from HoC[neg], keep only CIViC-like in HoC[pos]
    percentile: remove percentiles (CIViC-like from HoC[neg], HoC[neg]-like from HoC[pos], CIViC-unlike from HoC[pos)
    """

    if mode == "percentile":
        # Remove given best/worst percentile of sentences from each set

        print("\nRemoving CIViC-like sentences from HoC[neg] (", percentiles[0], "%)\n")
        hocneg_ = remove_most_similar_percent(U=hocneg, P=civic, ratio=ratio, percentile=percentiles[0])

        print("\nRemoving HoC[neg]-like sentences from HoC[pos] (", percentiles[1], "%)\n")
        hocpos_ = remove_most_similar_percent(U=hocpos, P=hocneg_, ratio=ratio, percentile=percentiles[1])

        print("\nRemoving CIViC-unlike sentences from HoC[pos] (", percentiles[2], "%)\n")
        hocpos_ = remove_most_similar_percent(U=hocpos_, P=civic, ratio=ratio, percentile=percentiles[2],
                                              inverse=True)

    elif mode == "strict":
        # Remove "good" sentences from HoC[neg], keep only "good" sentences in HoC[pos]
        # (tends to overfit on differences between Civic + Abstracts VS. HoC)

        print("\nRemoving CIViC-like sentences from HoC[neg]\n")
        hocneg_ = remove_P_from_U(P=civic, U=hocneg, ratio=ratio)

        print("\nKeeping only CIViC-like sentences in HoC[pos]\n")
        hocpos_ = remove_P_from_U(P=civic, U=hocpos, ratio=ratio, inverse=True)

    else:
        # Remove "good" sentences from HoC[neg], remove "bad" sentences in HoC[pos]
        # (appears to be most reasonable choice)

        print("\nRemoving CIViC-like sentences from HoC[neg]\n")
        hocneg_ = remove_P_from_U(P=civic, U=hocneg, ratio=ratio)

        print("\nRemoving HoC[neg]-like sentences from HoC[pos]\n")
        hocpos_ = remove_P_from_U(P=hocneg_, U=hocpos, ratio=ratio)

    P_raw = np.concatenate((hocpos_, civic))
    U_raw = abstracts
    N_raw = hocneg_

    if ratio < 1.0:
        P_raw, _ = train_test_split(P_raw, train_size=ratio)
        N_raw, _ = train_test_split(N_raw, train_size=ratio)
        U_raw, _ = train_test_split(U_raw, train_size=ratio)

    return P_raw, N_raw, U_raw


def vectorized_clean_pnu(ratio=1.0):
    P_raw, N_raw, U_raw = clean_corpus_pnu(ratio)

    print("\nSEMI-SUPERVISED TRAINING", "(on", 100 * ratio, "% of available data)",
          "\tP: HOC POS + CIVIC (", num_rows(P_raw), ")",
          "\tN: HOC NEG (", num_rows(N_raw), ")",
          "\tU: ABSTRACTS (", num_rows(U_raw), ")"
          )

    vec = transformers.vectorizer(genia_opts=genia_defaults)
    vec.fit(np.concatenate((P_raw, N_raw, U_raw)))

    P = vec.transform(P_raw)
    N = vec.transform(N_raw)
    U = vec.transform(U_raw)

    print("Features before selection:", np.shape(P)[1])

    sel = transformers.percentile_selector()
    sel.fit(vstack((P, N, U)),
            (np.concatenate((np.ones(num_rows(P)), -np.ones(num_rows(N)), np.zeros(num_rows(U))))))

    P = densify(sel.transform(P))
    N = densify(sel.transform(N))
    U = densify(sel.transform(U))

    print("Features after selection:", np.shape(P)[1])

    return P, N, U, vec, sel


# ------------------
# Cleanup for PU learning: obsolete / only used in test script
# ------------------

def clean_corpus_pu(ratio=1.0):
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
    hocneg_ = remove_P_from_U(U=hocneg, P=civic, ratio=ratio)

    print("\nRemoving HoC[neg]-like sentences from HoC[pos]\n")
    hocpos_ = remove_P_from_U(U=hocpos, P=hocneg_, ratio=ratio)

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

    return P_raw, U_raw, X_test_raw, y_test


def vectorized_clean_pu(ratio=1.0):
    P_raw, U_raw, X_test_raw, y_test = clean_corpus_pu(ratio)

    print("\nPU TRAINING", "(on", 100 * ratio, "% of available data)",
          "\tP: HOC POS + CIVIC", "(", num_rows(P_raw), ")",
          "\tN: HOC NEG + ABSTRACTS (", num_rows(U_raw), ")",
          "\tTEST SET (HOC POS + CIVIC + HOC NEG):", num_rows(X_test_raw)
          )

    vec = transformers.vectorizer(genia_opts=genia_defaults)
    vec.fit(np.concatenate((P_raw, U_raw)))

    P = vec.transform(P_raw)
    U = vec.transform(U_raw)

    print("Features before selection:", np.shape(P)[1])

    # sel = IdentitySelector()
    sel = transformers.percentile_selector()
    # sel = basic_pipeline.factorization('LatentDirichletAllocation')

    sel.fit(vstack((P, U)),
            (np.concatenate((np.ones(num_rows(P)), np.zeros(num_rows(U))))))
    P = densify(sel.transform(P))
    U = densify(sel.transform(U))
    X_test = densify(sel.transform(vec.transform(X_test_raw)))

    print("Features after selection:", np.shape(P)[1])

    return P, U, X_test, y_test, vec, sel
