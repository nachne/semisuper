import datetime
import multiprocessing as multi
import os
import pickle
import time
from copy import deepcopy
from functools import partial
from itertools import product

import numpy as np
from scipy.sparse import vstack
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

import semisuper.ss_techniques as ss
from semisuper import transformers
from semisuper.helpers import num_rows, densify

PARALLEL = False  # TODO multiprocessing works on Linux when there aren't too many features, but not on macOS


def best_model_cross_val(P, N, U, fold=10):
    """determine best model, cross validate and return pipeline trained on all data"""

    print("\nFinding best model\n")

    best = get_best_model(P, N, U)['best']

    print("\nCross-validation\n")

    kf = KFold(n_splits=fold, shuffle=True)
    splits = zip(list(kf.split(P)), list(kf.split(N)))

    if PARALLEL:
        with multi.Pool(fold) as p:
            stats = list(p.map(partial(eval_fold, best, P, N, U), enumerate(splits), chunksize=1))
    else:
        stats = list(map(partial(eval_fold, best, P, N, U), enumerate(splits)))

    mean_stats = np.mean(stats, 0)
    print("Cross-validation average: p {}, r {}, f1 {}, acc {}".format(
            mean_stats[0], mean_stats[1], mean_stats[2], mean_stats[3]))

    print("Retraining model on full data")

    vec, sel = best['vectorizer'], best['selector']
    vec.fit(np.concatenate((P, N, U)))
    P_, N_, U_ = [vec.transform(x) for x in [P, N, U]]

    y_pp = np.concatenate((np.ones(num_rows(P)), -np.ones(num_rows(N)), np.zeros(num_rows(U))))
    sel.fit(vstack((P_, N_, U_)), y_pp)
    P_, N_, U_ = [densify(sel.transform(x)) for x in [P_, N_, U_]]

    model = best['untrained_model'](P_, N_, U_)

    print("Ratio of U classified as positive:", np.sum(model.predict(U_)) / num_rows(U_))
    print("Returning final model")

    return Pipeline([('vectorizer', vec), ('selector', sel), ('clf', model)])


# helper
def eval_fold(model_record, P, N, U, i_splits):
    """helper function for running cross validation in parallel"""

    i, (p_split, n_split) = i_splits
    P_train, P_test = P[p_split[0]], P[p_split[1]]
    N_train, N_test = N[n_split[0]], N[n_split[1]]

    y_train_pp = np.concatenate((np.ones(num_rows(P_train)), -np.ones(num_rows(N_train)), np.zeros(num_rows(U))))
    pp = Pipeline([('vectorizer', model_record['vectorizer']), ('selector', model_record['selector'])])
    pp.fit(np.concatenate((P_train, N_train, U)), y_train_pp)
    P_, N_, U_, P_test_, N_test_ = [densify(pp.transform(x)) for x in [P_train, N_train, U, P_test, N_test]]

    model = model_record['untrained_model'](P_, N_, U_)

    y_pred = model.predict(np.concatenate((P_test_, N_test_)))
    y_test = np.concatenate((np.ones(num_rows(P_test_)), np.zeros(num_rows(N_test_))))

    pr, r, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("Fold no.", i, "acc", acc, "classification report:\n", classification_report(y_test, y_pred))
    return [pr, r, f1, acc]


def get_best_model(P_train, N_train, U_train, X_test=None, y_test=None):
    """Evaluate parameter combinations, save results and return object with stats of all models"""

    print("\nEvaluating parameter ranges for preprocessor and classifiers")

    if X_test is None or y_test is None:
        P_train, X_test_pos = train_test_split(P_train, test_size=0.2)
        N_train, X_test_neg = train_test_split(N_train, test_size=0.2)
        X_test = np.concatenate((X_test_pos, X_test_neg))
        y_test = np.concatenate((np.ones(num_rows(X_test_pos)), np.zeros(num_rows(X_test_neg))))

    X_eval, X_dev, y_eval, y_dev = train_test_split(X_test, y_test, test_size=0.75)

    X_train = np.concatenate((P_train, N_train, U_train))
    y_train_pp = np.concatenate((np.ones(num_rows(P_train)), -np.ones(num_rows(N_train)), np.zeros(num_rows(U_train))))

    results = {'best': {'f1': -1, 'acc': -1}, 'all': []}

    preproc_params = {
        'df_min'        : [0.001],
        'df_max'        : [1.0],
        'rules'         : [True],  # [True, False],
        'lemmatize'     : [False],
        'wordgram_range': [(1, 4)],  # [(1, 2), (1, 3), (1, 4)],  # [None, (1, 2), (1, 3), (1, 4)],
        'chargram_range': [(2, 6)],  # [None, (2, 4), (2, 5), (2, 6)],
        'feature_select': [
            # best: word (1,2)/(1,4), char (2,5)/(2,6), f 25%, rule True/False, SVC 1.0 / 0.75
            # w/o char: acc <= 0.80, w/o words: acc <= 0.84, U > 34%

            # partial(basic_pipeline.percentile_selector, 'chi2', 30),
            partial(transformers.percentile_selector, 'chi2', 25),
            # partial(basic_pipeline.percentile_selector, 'chi2', 20),
            # partial(basic_pipeline.percentile_selector, 'f', 30),
            # partial(basic_pipeline.percentile_selector, 'f', 25),
            # partial(basic_pipeline.percentile_selector, 'f', 20),
            # partial(basic_pipeline.percentile_selector, 'mutual_info', 30), # mutual information: worse than rest
            # partial(basic_pipeline.percentile_selector, 'mutual_info', 25),
            # partial(basic_pipeline.percentile_selector, 'mutual_info', 20),
            # partial(basic_pipeline.factorization, 'TruncatedSVD', 1000),
            # partial(basic_pipeline.factorization, 'TruncatedSVD', 2000), # 10% worse than chi2, slow, SVM iter >100
            # partial(basic_pipeline.factorization, 'TruncatedSVD', 3000),
        ]
    }

    for wordgram, chargram in product(preproc_params['wordgram_range'], preproc_params['chargram_range']):
        for r, l in product(preproc_params['rules'], preproc_params['lemmatize']):
            for df_min, df_max in product(preproc_params['df_min'], preproc_params['df_max']):
                for fs in preproc_params['feature_select']:

                    if wordgram is None and chargram is None:
                        break

                    print("\n----------------------------------------------------------------",
                          "\nwords:", wordgram, "chars:", chargram, "feature selection:", fs, "df_min:", df_min,
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
                        {'name': 'neglinSVC_C1.0', 'model': partial(ss.iterate_linearSVC_C, 1.0)},
                        # {'name': 'neglinSVC_C.75', 'model': partial(ss.iterate_linearSVC_C, 0.75)},
                        # {'name': 'neglinSVC_C0.5', 'model': partial(ss.iterate_linearSVC_C, 0.5)},
                        # {'name' : 'negSGDmh',
                        #  'model': partial(ss.neg_self_training_clf, SGDClassifier(loss='modified_huber'))},
                        # {'name' : 'negSGDsh',
                        #  'model': partial(ss.neg_self_training_clf, SGDClassifier(loss='squared_hinge'))},
                        # {'name' : 'negSGDpc',
                        #  'model': partial(ss.neg_self_training_clf, SGDClassifier(loss='perceptron'))},
                        # {'name': 'negNB0.1', 'model': partial(ss.neg_self_training_clf, MultinomialNB(alpha=0.1))},
                        # {'name': 'negNB1.0', 'model': partial(ss.neg_self_training_clf, MultinomialNB(alpha=1.0))},
                        # {'name' : 'self-logit', 'model': ss.self_training},
                        # {'name' : 'EM', 'model': ss.EM},
                        # {'name' : 'kNN', 'model': ss.iterate_knn},
                        # {'name' : 'label_propagation', 'model': ss.propagate_labels},
                    ]

                    if PARALLEL:
                        with multi.Pool(min(multi.cpu_count(), len(iteration))) as p:
                            iter_stats = list(p.map(partial(model_eval_record,
                                                            P_train_, N_train_, U_train_, X_dev_, y_dev),
                                                    iteration, chunksize=1))
                    else:
                        iter_stats = list(map(partial(model_eval_record,
                                                      P_train_, N_train_, U_train_, X_dev_, y_dev),
                                              iteration))

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

    return test_best(results, X_eval, y_eval)


def test_best(results, X_eval, y_eval):
    """helper function to evaluate best model on held-out set. returns full results of model selection"""

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
    """prepare training and test vectors, vectorizer and selector for validating classifiers"""

    print("Fitting vectorizer, preparing training and test data")

    vectorizer = transformers.vectorizer(chargrams=chargram_range, min_df_char=min_df_char, wordgrams=wordgram_range,
                                         min_df_word=min_df_word, max_df=max_df, lemmatize=lemmatize, rules=rules)

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


def model_eval_record(P, N, U, X, y, m):
    """helper function for finding best model in parallel: evaluate model and return stat object. """

    untrained_model = m['model']
    model = m['model'](P, N, U)
    name = m['name']

    y_pred = model.predict(X)

    p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average='macro')
    acc = accuracy_score(y, y_pred)
    clsr = classification_report(y, y_pred)

    pos_ratio = np.sum(model.predict(U)) / num_rows(U)

    print("\n")
    # print("\n{}:\tacc: {}, classification report:\n{}".format(name, acc, clsr))
    return {'name' : name, 'p': p, 'r': r, 'f1': f1, 'acc': acc, 'clsr': clsr,
            'model': model, 'untrained_model': untrained_model, 'U_ratio': pos_ratio}


def print_results(results):
    """helper function to print stat objects, starting with best model"""

    print("\n----------------------------------------------------------------\n")
    print("\nAll stats:\n")
    for i in results['all']:
        print_reports(i)

    print("\nBest:\n")
    best = results['best']
    print(best['name'], best['n-grams'], best['fs'],
          "\nrules, lemma", best['rules, lemma'], "df_min, df_max", best['df_min, df_max'],
          "\namount of U labelled as relevant:", best['U_ratio'],
          "\nstats: p={}\tr={}\tf1={}\tacc={}\t".format(best['p'], best['r'], best['f1'], best['acc']))

    print("\n----------------------------------------------------------------\n")
    return


def print_reports(i):
    """helper to print model stat object"""
    print(i[0]['n-grams'], i[0]['fs'],
          "\nrules, lemma", i[0]['rules, lemma'], "df_min, df_max", i[0]['df_min, df_max'])

    for m in i:
        print("\n{}:\tacc: {}, relevant ratio in U: {}, classification report:\n{}".format(
                m['name'], m['acc'], m['U_ratio'], m['clsr']))
    return


def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)