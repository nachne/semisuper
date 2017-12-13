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

from semisuper import transformers
from semisuper.helpers import num_rows, densify

PARALLEL = False  # TODO multiprocessing works on Linux when there aren't too many features, but not on macOS




def get_best_model(X_train, y_train, weights=None, X_test=None, y_test=None):
    """Evaluate parameter combinations, save results and return object with stats of all models"""

    print("\nEvaluating parameter ranges for preprocessor and classifiers")

    if weights is None:
        weights = np.ones(num_rows(X_train))

    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test, weights_train, _ = train_test_split(X_train, y_train, weights,
                                                                              test_size=0.2)

    X_eval, X_dev, y_eval, y_dev = train_test_split(X_test, y_test, test_size=0.5)

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
                                                                                trainLabels=y_train,
                                                                                wordgram_range=wordgram,
                                                                                chargram_range=chargram,
                                                                                min_df_char=df_min,
                                                                                min_df_word=df_min,
                                                                                max_df=df_max,
                                                                                feature_select=fs,
                                                                                lemmatize=l,
                                                                                rules=r)

                    # fit models
                    iteration = [
                        {"name": "MNB", "model": MultinomialNB()},
                        {"name": "SGD", "model": SGDClassifier(loss='modified_huber')},
                        {"name": "SVC", "model": LinearSVC()},
                    ]

                    if PARALLEL:
                        with multi.Pool(min(multi.cpu_count(), len(iteration))) as p:
                            iter_stats = list(p.map(partial(model_eval_record,
                                                            X_train_, y_train, weights_train, X_dev_, y_dev),
                                                    iteration, chunksize=1))
                    else:
                        iter_stats = list(map(partial(model_eval_record,
                                                      X_train_, y_train, weights_train, X_dev_, y_dev),
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

                    # print_reports(iter_stats)

    # print_results(results)

    return test_best(results, X_eval, y_eval)


def model_eval_record(X_train, y_train, weights_train, X_test, y_test, m):
    """helper function for finding best model in parallel: evaluate model and return stat object. """

    model = m['model'].fit(X_train, y_train, sample_weight=weights_train)
    name = m['name']

    y_pred = model.predict(X_test)

    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    clsr = classification_report(y_test, y_pred)

    print("\n")
    print("\n{}:\tacc: {}, classification report:\n{}".format(name, acc, clsr))
    return {'name' : name, 'p': p, 'r': r, 'f1': f1, 'acc': acc, 'clsr': clsr,
            'model': model}


def test_best(results, X_eval, y_eval):
    """helper function for finding best model in parallel: evaluate model and return stat object. """

    best_model = results['best']['model']
    name = results['best']['name']

    selector = results['best']['selector']
    vectorizer = results['best']['vectorizer']

    if selector:
        transformedTestData = densify(selector.transform(vectorizer.transform(X_eval)))
    else:
        transformedTestData = densify(vectorizer.transform(X_eval))

    y_pred = best_model.predict(transformedTestData)

    p, r, f1, _ = precision_recall_fscore_support(y_eval, y_pred, average='macro')
    acc = accuracy_score(y_eval, y_pred)
    clsr = classification_report(y_eval, y_pred)

    print("\n")
    print("\n{}:\tacc: {}, classification report:\n{}".format(name, acc, clsr))

    return Pipeline([('vectorizer', vectorizer), ('selector', selector), ('clf', best_model)])


def prepare_train_test(trainData, testData, trainLabels, rules=True, wordgram_range=None, feature_select=None,
                       chargram_range=None, lemmatize=True, min_df_char=0.001, min_df_word=0.001, max_df=1.0):
    """prepare training and test vectors, vectorizer and selector for validating classifiers"""

    print("Fitting vectorizer, preparing training and test data")

    transformers.vectorizer()

    vectorizer = transformers.vectorizer_dx(chargrams=chargram_range, min_df_char=min_df_char, wordgrams=wordgram_range,
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
