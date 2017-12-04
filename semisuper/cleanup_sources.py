import multiprocessing as multi
import random
from functools import partial

import numpy as np
from scipy.sparse import vstack
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC

from semisuper import loaders, pu_two_step, basic_pipeline, pu_cos_roc, pu_biased_svm
from semisuper.helpers import num_rows, densify, pu_score, select_PN_below_score

civic, abstracts = loaders.sentences_civic_abstracts()
hocpos, hocneg = loaders.sentences_HoC()
piboso_other = loaders.sentences_piboso_other()
piboso_outcome = loaders.sentences_piboso_outcome()


# ------------------
# select sentences
# ------------------

def remove_P_from_U(noisy, guide, ratio=1.0, inverse=False, verbose=True):
    """Remove sentences from noisy_set that are similar to guide_set according to strictest PU estimator.

    if inverse is set to True, keep rather than discard them."""

    guide_, noisy_, vectorizer, selector = prepare_pu(guide, noisy, ratio=ratio)

    model = best_pu(guide_, noisy_)

    y_noisy = model.predict(selector.transform(densify(vectorizer.transform(noisy))))

    if inverse:
        action = "Keeping"
        criterion = 1
    else:
        action = "Discarding"
        criterion = 0

    print(action, (100 * np.sum(y_noisy) / num_rows(y_noisy)), "% of noisy data (", np.sum(y_noisy), "sentences )",
          "as per result of PU learning")

    keeping = np.array([x for (x, y) in zip(noisy, y_noisy) if y == criterion])

    if verbose:
        discarding = [x for (x, y) in zip(noisy, y_noisy) if y != criterion]
        print("Keeping", random.sample(keeping.tolist(), 10))
        print("Discarding", random.sample(discarding, 10))

    return keeping


def remove_least_similar_percent(noisy, guide, ratio=1.0, percentile=10, inverse=False, verbose=True):
    """Remove percentile of sentences from noisy_set that are similar to guide_set according to strictest PU estimator.

    if inverse is set to True, remove least rather than most similar."""

    guide_, noisy_, vectorizer, selector = prepare_pu(guide, noisy, ratio=ratio)

    model = pu_cos_roc.ranking_cos_sim(guide_)

    if inverse:
        predicate = "least"
        y_pred = model.predict_proba(noisy_)
    else:
        predicate = "most"
        y_pred = -model.predict_proba(noisy_)

    print("Removing", percentile, "% of noisy data", predicate, "similar to guide set (cos-similarity)"
          , "(", (percentile * num_rows(noisy) / 100), "sentences )")

    U = np.array(noisy)
    U_minus_PN, PN = select_PN_below_score(y_pred, U, y_pred, noise_lvl=percentile / 100)

    if verbose:
        print("Keeping", random.sample(U_minus_PN, 10))
        print("Discarding", random.sample(PN, 10))

    return U_minus_PN


# ------------------
# model selection
# ------------------

def best_pu(P, U):
    P_train, P_test = train_test_split(P, test_size=0.2)
    U_train, U_test = train_test_split(U, test_size=0.2)

    models = [
        # {"name": "I-EM", "model": partial(pu_two_step.i_EM, P_train, U_train)},
        # {"name": "S-EM", "model": partial(pu_two_step.s_EM, P_train, U_train)},
        # {"name": "ROC-EM", "model": partial(pu_two_step.roc_EM, P_train, U_train)},
        {"name": "ROC-SVM", "model": partial(pu_two_step.roc_SVM, P_train, U_train)},
        {"name": "CR-SVM", "model": partial(pu_two_step.cr_SVM, P_train, U_train)},
        # {"name": "SPY-SVM", "model": partial(pu_two_step.spy_SVM, P_train, U_train)},
        # {"name": "ROCCHIO", "model": partial(pu_two_step.rocchio, P_train, U_train)},
        # {"name": "BIASED-SVM", "model": partial(pu_biased_svm.biased_SVM_weight_selection, P_train, U_train)},
        # {"name": "supervised SVC", "model": partial(LinearSVC().fit,
        #                                             np.concatenate((P_train, U_train)),
        #                                             np.concatenate((np.ones(num_rows(P_train)),
        #                                                             np.zeros(num_rows(U_train)))))}
    ]

    with multi.Pool(min(len(models), multi.cpu_count() // 4)) as p:
        stats = list(map(partial(model_pu_score_record, P_test, U_test), models))

    for s in stats:
        print(s["name"], "\tPU-score:", s["pu_score"])

    best_model = max(stats, key=lambda x: x["pu_score"])
    print("Best PU model:", best_model["name"], "\tPU-score:", best_model["pu_score"])

    return best_model["model"]


def model_pu_score_record(P_test, U_test, m):
    model = m['model']()
    name = m['name']

    y_P = model.predict(P_test)
    y_U = model.predict(U_test)

    score = pu_score(y_P, y_U)

    return {'name': name, 'model': model, 'pu_score': score}


# ------------------
# preprocess
# ------------------


def prepare_pu(P, U, ratio=1.0):
    """generate and select features for ratio of sentence sets"""

    print("Preprocessing corpora for PU learning")
    print("Training on", 100 * ratio, "% of data")
    if ratio < 1.0:
        P, _ = train_test_split(P, train_size=ratio)
        U, _ = train_test_split(U, train_size=ratio)

    # wordgram_range = (1, 4)  # TODO change back to (1,3)
    # chargram_range = (2, 6)  # TODO change back to (3,6)
    # min_df_word, min_df_char = [20, 20]
    # rules, lemmatize = [True, True]
    #
    # def print_params():
    #     print("word n-gram range:", wordgram_range,
    #           "\nchar n-gram range:", chargram_range,
    #           "\nmin_df: word", min_df_word, "char:", min_df_char,
    #           "\nrule-based preprocessing:", rules, "\tlemmatization:", lemmatize)
    #     return
    #
    # print_params()
    #
    # print("Fitting vectorizer")
    # vec = basic_pipeline.vectorizer(wordgrams=wordgram_range,
    #                                 chargrams=chargram_range, rules=rules, lemmatize=lemmatize)
    vec = basic_pipeline.vectorizer()
    vec.fit(np.concatenate((P, U)))

    P_ = vec.transform(P)
    U_ = vec.transform(U)

    print("Features before selection:", np.shape(U_)[1])

    # TODO FIXME choose best selector
    # sel = identitySelector()
    # sel = basic_pipeline.factorization()
    # sel = basic_pipeline.percentile_selector()
    sel = basic_pipeline.factorization('LatentDirichletAllocation')
    sel.fit(vstack((P_, U_)),
            np.concatenate((np.ones(num_rows(P_)), -np.ones(num_rows(U_)))))

    # TODO remove
    print(sel.perplexity(vstack((P, U))))

    P_ = densify(sel.transform(P_))
    U_ = densify(sel.transform(U_))

    return P_, U_, vec, sel

# TODO obsolete
def prepare_corpus(ratio=0.5):
    hocpos_train, hocpos_test = train_test_split(hocpos, test_size=0.2)

    hocneg_train, hocneg_test = train_test_split(hocneg, test_size=0.2)
    civic_train, civic_test = train_test_split(civic, test_size=0.2)
    abstracts_train, abstracts_test = train_test_split(abstracts, test_size=0.2)

    if ratio < 1.0:
        hocpos_train = random.sample(hocpos_train, int(ratio * num_rows(hocpos_train)))
        hocneg_train = random.sample(hocneg_train, int(ratio * num_rows(hocneg_train)))
        civic_train = random.sample(civic_train, int(ratio * num_rows(civic_train)))
        abstracts_train = random.sample(abstracts_train, int(ratio * num_rows(abstracts_train)))

    wordgram_range = (1, 4)  # TODO change back to True, (1,3)
    chargram_range = (2, 6)  # TODO change back to True, (3,6)
    min_df_word, min_df_char = [20, 20]
    rules, lemmatize = [True, True]

    def print_params():
        print("word n-gram range:", wordgram_range,
              "\nchar n-gram range:", chargram_range,
              "\nmin_df: word", min_df_word, "char:", min_df_char,
              "\nrule-based preprocessing:", rules, "\tlemmatization:", lemmatize)
        return

    print_params()

    print("Fitting vectorizer")
    vec = basic_pipeline.vectorizer(chargrams=chargram_range, wordgrams=wordgram_range, lemmatize=lemmatize,
                                           rules=rules)
    vec.fit(np.concatenate((civic_train, hocpos_train, hocneg_train, abstracts_train)))

    hocpos_train = densify(vec.transform(hocpos_train))
    hocneg_train = densify(vec.transform(hocneg_train))
    civic_train = densify(vec.transform(civic_train))
    abstracts_train = densify(vec.transform(abstracts_train))

    print("Features before selection:", np.shape(hocpos_train)[1])

    # selector = identitySelector()
    sel = basic_pipeline.percentile_selector(percentile=20)
    sel = basic_pipeline.factorization('LatentDirichletAllocation', n_components=10)
    sel.fit(np.concatenate((civic_train, hocneg_train, hocpos_train, abstracts_train)),
                 (np.concatenate((np.ones(num_rows(civic_train)),
                                  -np.ones(num_rows(hocneg_train)),
                                  np.zeros(num_rows(hocpos_train)),
                                  2 * np.ones(num_rows(abstracts_train))))))

    # TODO adjust classes to use as P, N, U
    P = densify(sel.transform(civic_train))
    P_test = densify(sel.transform(densify(vec.transform(civic_test))))

    N = densify(sel.transform(hocpos_train))
    N_test = densify(sel.transform(densify(vec.transform(hocpos_test))))

    U = densify(sel.transform(hocneg_train))

    print("\nPURIFYING SOURCES SEMI-SUPERVISED (", ratio, "% of data )"
          , "\tHOC POS", "(N)"
          , "(", num_rows(hocpos_train), ")"
          , "\tHOC NEG", "(U)"
          , "(", num_rows(hocneg_train), ")"
          , ",\tCIVIC", "(P)"
          , "(", num_rows(civic_train), ")"
          , "\tABSTRACTS"  # , "(N)"
          , "(", num_rows(abstracts_train), ")"
          )

    X_test = np.concatenate((P_test, N_test))
    y_test = np.concatenate((np.ones(num_rows(P_test)), np.zeros(num_rows(N_test))))

    # print("Features after selection:", np.shape(P)[1])

    return P, N, U, X_test, y_test, vec, sel
