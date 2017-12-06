import datetime
import multiprocessing as multi
import os
import pickle
import time
from copy import deepcopy
from functools import partial
from itertools import product

import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline

import semisuper.basic_pipeline as basic_pipeline
import semisuper.ss_techniques as ss
from semisuper.helpers import num_rows, densify


def best_model_cross_val(P, N, U):
    print("\nFinding best model\n")

    best = get_best_model(P, N, U)['best']

    print("\nCross-validation\n")

    kf = KFold(n_splits=5)
    splits = [list(kf.split()), list(kf.split(N))]

    stats = []

    for i, (p_split, n_split) in enumerate(splits):
        P_train, P_test = P[p_split[0]], P[p_split[1]]
        N_train, N_test = N[n_split[0]], N[n_split[1]]

        y_train_pp = np.concatenate((np.ones(num_rows(P_train)),
                                     -np.ones(num_rows(N_train)),
                                     np.zeros(num_rows(U))))
        pp = Pipeline([('vectorizer', best['vectorizer']), ('selector', best['selector'])])
        pp.fit(np.concatenate((P_train, N_train, U), y_train_pp))
        P_, N_, U_, P_test_, N_test_ = [pp.transform(x) for x in [P_train, N_train, U, P_test, N_test]]

        model = best['untrained_model'](P_, N_, U_)
        y_pred = model.predict(np.concatenate((P_test_, N_test_)))
        y_test = np.concatenate((np.ones(num_rows(P_test_)), np.zeros(num_rows(N_test_))))

        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        stats.append([p, r, f1, acc])

        print("Fold no.", i, "acc", acc, classification_report(y_test, y_pred))

    mean_stats = np.mean(stats, 0)
    print("Cross-validation average: p {}, r {}, f1 {}, acc {}".format(
            mean_stats[0], mean_stats[1], mean_stats[2], mean_stats[3]))

    print("Retraining model on full data")

    vec, sel = best['vectorizer'], best['selector']
    vec.fit(np.concatenate((P, N, U)))
    P_, N_, U_ = [vec.transform(x) for x in [P, N, U]]

    y_pp = np.concatenate((np.ones(num_rows(P)), -np.ones(num_rows(N)), np.zeros(num_rows(U))))
    sel.fit(np.concatenate((P_, N_, U_)), y_pp)
    P_, N_, U_ = [densify(sel.transform(x)) for x in [P_, N_, U_]]

    model = best['untrained_model'](P_, N_, U_)

    print("Ratio of U classified as positive:", np.sum(model.predict(U_)) / num_rows(U_))
    print("Returning final model")

    return Pipeline([('vectorizer', vec),
                     ('selector', sel),
                     ('clf', model)])


def get_best_model(P_train, N_train, U_train, X_test=None, y_test=None):
    """Evaluate parameter combinations, save results and return pipeline with best model"""

    print("\nEvaluating parameter ranges for preprocessor and classifiers")

    if X_test is None or y_test is None:
        P_train, X_test_pos = train_test_split(P_train, test_size=0.2)
        N_train, X_test_neg = train_test_split(N_train, test_size=0.2)
        X_test = np.concatenate((X_test_pos, X_test_neg))
        y_test = np.concatenate((np.ones(num_rows(X_test_pos)), np.zeros(X_test_neg)))

    X_eval, X_dev, y_eval, y_dev = train_test_split(X_test, y_test, test_size=0.5)

    X_train = np.concatenate((P_train, N_train, U_train))
    y_train_pp = np.concatenate((np.ones(num_rows(P_train)),
                                 -np.ones(num_rows(N_train)),
                                 np.zeros(num_rows(U_train))))

    results = {'best': {'f1': -1, 'acc': -1}, 'all': []}

    preproc_params = {
        'df_min'        : [0.002],
        'df_max'        : [1.0],
        'rules'         : [True],
        'lemmatize'     : [False],
        'wordgram_range': [(1, 4)],  # [None, (1, 2), (1, 3), (1, 4)],
        'chargram_range': [(2, 6)],  # [None, (2, 4), (2, 5), (2, 6)],
        'feature_select': [partial(basic_pipeline.percentile_selector, 'chi2', 30),
                           # partial(basic_pipeline.percentile_selector, 'chi2', 20),
                           # partial(basic_pipeline.percentile_selector, 'f', 20),
                           # partial(basic_pipeline.percentile_selector, 'mutual_info', 20),
                           # partial(basic_pipeline.factorization, 'TruncatedSVD', 1000),
                           ]
    }

    for wordgram, chargram in product(preproc_params['wordgram_range'], preproc_params['chargram_range']):
        for r, l in product(preproc_params['rules'], preproc_params['lemmatize']):
            for df_min, df_max in product(preproc_params['df_min'], preproc_params['df_max']):
                for fs in preproc_params['feature_select']:

                    if wordgram == None and chargram == None:
                        break

                    print("\n----------------------------------------------------------------",
                          "\nwords:", wordgram, "chars:", chargram, "feature selection:", fs,
                          "\n----------------------------------------------------------------\n")

                    start_time = time.time()

                    X_train_, X_dev_, vectorizer, selector = prepare_train_test(trainData=X_train, testData=X_dev,
                                                                                trainLabels=y_train_pp,
                                                                                wordgram_range=wordgram,
                                                                                chargram_range=chargram,
                                                                                min_df_char=df_min,
                                                                                min_df_word=df_min,
                                                                                max_df=df_max,
                                                                                feature_select=fs,
                                                                                lemmatize=l,
                                                                                rules=r)
                    if selector:
                        P_train_, N_train_, U_train_ = [densify(selector.transform(vectorizer.transform(x)))
                                                        for x in [P_train, N_train, U_train]]
                    else:
                        P_train_, N_train_, U_train_ = [densify(vectorizer.transform(x))
                                                        for x in [P_train, N_train, U_train]]

                    # fit models
                    iteration = [
                        # {'name'           : 'neg-linSVC_C1.0',
                        #  'model'          : partial(ss.iterate_linearSVC, P_train_, N_train_, U_train_, 1.0),
                        #  'untrained_model': ss.iterate_linearSVC},
                        {'name' : 'neg-linSVC_C0.5',
                         'model': partial(ss.iterate_linearSVC, P_train_, N_train_, U_train_, 0.5),
                         'untrained_model': ss.iterate_linearSVC},
                        # {'name'           : 'neg-SGD',
                        #  'model'          : partial(ss.neg_self_training_sgd, P_train_, N_train_, U_train_),
                        #  'untrained_model': ss.neg_self_training_sgd},
                        # {'name'           : 'self-logit',
                        #  'model'          : partial(ss.self_training, P_train_, N_train_, U_train_),
                        #  'untrained_model': ss.self_training},
                    ]

                    # eval models
                    # TODO multiprocessing; breaks on macOS but not on Linux
                    with multi.Pool(min(multi.cpu_count(), len(iteration))) as p:
                        iter_stats = list(map(partial(model_eval_record, X_dev_, y_dev, U_train_), iteration))

                    # finalize records: remove model, add n-gram stats, update best
                    for m in iter_stats:
                        m['n-grams'] = {'word': wordgram, 'char': chargram},
                        m['rules, lemma'] = (r, l)
                        m['df_min, df_max'] = (df_min, df_max)
                        m['fs'] = fs()
                        if m['acc'] > results['best']['acc']:
                            results['best'] = deepcopy(m)
                            results['best']['vectorizer'] = vectorizer
                            results['best']['selector'] = selector
                        m.pop('model', None)

                    results['all'].append(iter_stats)

                    print("Evaluated words:", wordgram, "chars:", chargram,
                          "in %s seconds\n" % (time.time() - start_time))

                    print_reports(iter_stats)

    print_results(results)

    # ----------------------------------------------------------------
    # test best on held-out test set
    # ----------------------------------------------------------------

    best_model = results['best']['model']
    selector = results['best']['selector']
    vectorizer = results['best']['vectorizer']

    if selector:
        transformedTestData = densify(selector.transform(vectorizer.transform(X_eval)))
    else:
        transformedTestData = densify(vectorizer.transform(X_eval))

    y_pred_test = best_model.predict(transformedTestData)

    p, r, f, s = precision_recall_fscore_support(y_eval, y_pred_test, average='macro')
    acc = accuracy_score(y_eval, y_pred_test)

    print("Testing best model on held-out test set:\n", results['best']['name'],
          results['best']['n-grams'], results['best']['fs'], "\n",
          'p={}\tr={}\tf1={}\tacc={}'.format(p, r, f, acc))

    results['best']['eval stats'] = [p, r, f, s, acc]

    return results


def prepare_train_test(trainData, testData, trainLabels, rules=True, wordgram_range=None, feature_select=None,
                       chargram_range=None, lemmatize=True, min_df_char=0.001, min_df_word=0.001, max_df=1.0):
    """prepare training and test vectors and vectorizer for validating classifiers
    :param min_df_char:
    """

    print("Fitting vectorizer, preparing training and test data")

    vectorizer = basic_pipeline.vectorizer(chargrams=chargram_range, min_df_char=min_df_char, wordgrams=wordgram_range,
                                           min_df_word=min_df_word, lemmatize=lemmatize, rules=rules)

    transformedTrainData = vectorizer.fit_transform(trainData)
    transformedTestData = vectorizer.transform(testData)

    print("No. of features:", transformedTrainData.shape[1])

    selector = None
    if feature_select is not None:
        selector = feature_select()
        selector.fit(transformedTrainData, trainLabels)
        transformedTrainData = selector.transform(transformedTrainData)
        transformedTestData = selector.transform(transformedTestData)

        print("No. of features after reduction:", transformedTrainData.shape[1], "\n")
    print()
    return transformedTrainData, transformedTestData, vectorizer, selector


def model_eval_record(X, y, U, m):
    model = m['model']()
    name = m['name']

    y_pred = model.predict(X)

    p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average='macro')
    acc = accuracy_score(y, y_pred)
    clsr = classification_report(y, y_pred)

    pos_ratio = np.sum(model.predict(U)) / num_rows(U)

    print("\n")
    # print("\n{}:\tacc: {}, classification report:\n{}".format(name, acc, clsr))
    return {'name': name, 'p': p, 'r': r, 'f1': f1, 'acc': acc, 'clsr': clsr, 'model': model, 'U_ratio': pos_ratio}


def print_results(results):
    best = results['best']

    print("Best:")
    print(best['name'], best['n-grams'], best['fs'],
          "\nrules, lemma", best[0]['rules, lemma'], "df_min, df_max", best[0]['df_min, df_max'],
          "amount of U labelled as relevant:", best['U_ratio'],
          "\tstats: p={}\tr={}\tf1={}\tacc={}\t".format(best['p'], best['r'], best['f1'], best['acc']))

    print("All stats:")
    for i in results['all']:
        print_reports(i)
    return


def print_reports(i):
    print(i[0]['n-grams'], i[0]['fs'],
          "\nrules, lemma", i[0]['rules, lemma'], "df_min, df_max", i[0]['df_min, df_max'])

    for m in i:
        print("\n{}:\tacc: {}, relevant ratio in U: {}, classification report:\n{}".format(
                m['name'], m['acc'], m['U_ratio'], m['clsr']))
    return


def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)
