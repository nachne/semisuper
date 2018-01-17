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
from sklearn.model_selection import train_test_split

import semisuper.pu_biased_svm as biased_svm
import semisuper.pu_two_step as two_step
import semisuper.transformers as transformers
from semisuper.helpers import num_rows, concatenate

# TODO: obsolete (only used in test script, left here for documentation)
# TODO: cross-validation

# TODO multiprocessing; breaks on macOS but not on Linux
PARALLEL = True  # os.sys.platform == "linux"


def getBestModel(P_train, U_train, X_test, y_test):
    """Evaluate parameter combinations, save results and return pipeline with best model"""

    print("\nEvaluating parameter ranges for preprocessor and classifiers")

    X_train = concatenate((P_train, U_train))
    y_train_pp = concatenate((np.ones(num_rows(P_train)), np.zeros(num_rows(U_train))))

    results = {'best': {'f1': -1, 'acc': -1}, 'all': []}

    preproc_params = {
        'df_min'        : [0.002],
        'df_max'        : [1.0],
        'rules'         : [True],
        'genia_opts'    : [None],
        'wordgram_range': [(1, 4)],  # [None, (1, 2), (1, 3), (1, 4)],
        'chargram_range': [(2, 6)],  # [None, (2, 4), (2, 5), (2, 6)],
        'feature_select': [partial(transformers.percentile_selector, 'chi2'),
                           # partial(transformers.factorization, 'PCA', 10),
                           # partial(transformers.factorization, 'PCA', 100),
                           # partial(transformers.factorization, 'PCA', 1000),
                           ]
    }

    for wordgram, chargram in product(preproc_params['wordgram_range'], preproc_params['chargram_range']):
        for r, g in product(preproc_params['rules'], preproc_params['genia_opts']):
            for df_min, df_max in product(preproc_params['df_min'], preproc_params['df_max']):
                for fs in preproc_params['feature_select']:

                    if wordgram is None and chargram is None:
                        break

                    print("\n----------------------------------------------------------------",
                          "\nwords:", wordgram, "chars:", chargram, "feature selection:", fs,
                          "\n----------------------------------------------------------------\n")

                    start_time = time.time()

                    X_train_, X_dev_, vectorizer, selector = prepareTrainTest(trainData=X_train, testData=X_test,
                                                                              trainLabels=y_train_pp, rules=r,
                                                                              wordgram_range=wordgram,
                                                                              feature_select=fs,
                                                                              chargram_range=chargram, genia_opts=g,
                                                                              min_df_char=df_min, min_df_word=df_min,
                                                                              max_df=df_max)
                    if selector:
                        P_train_ = selector.transform(vectorizer.transform(P_train))
                        U_train_ = selector.transform(vectorizer.transform(U_train))
                    else:
                        P_train_ = vectorizer.transform(P_train)
                        U_train_ = vectorizer.transform(U_train)

                    pp = {'word': wordgram, 'char': chargram}

                    # fit models
                    iteration = [
                        {'name': 'i-em', 'model': partial(two_step.i_EM, P_train_, U_train_)},
                        {'name' : 's-em spy=0.1',
                         'model': partial(two_step.s_EM, P_train_, U_train_, spy_ratio=0.1, noise_lvl=0.1)},
                        {'name' : 's-em spy=0.2',
                         'model': partial(two_step.s_EM, P_train_, U_train_, spy_ratio=0.2, noise_lvl=0.2)},
                        {'name': 'roc-svm', 'model': partial(two_step.roc_SVM, P_train_, U_train_)},
                        {'name' : 'cr_svm noise=0.1',
                         'model': partial(two_step.cr_SVM, P_train_, U_train_, noise_lvl=0.1)},
                        {'name' : 'cr_svm noise=0.2',
                         'model': partial(two_step.cr_SVM, P_train_, U_train_, noise_lvl=0.2)},
                        {'name': 'roc_em', 'model': partial(two_step.roc_EM, P_train_, U_train_)},
                        {'name' : 'spy_svm spy=0.1',
                         'model': partial(two_step.spy_SVM, P_train_, U_train_, spy_ratio=0.1, noise_lvl=0.1)},
                        {'name' : 'spy_svm spy=0.2',
                         'model': partial(two_step.spy_SVM, P_train_, U_train_, spy_ratio=0.2, noise_lvl=0.2)},
                        {'name' : 'biased-svm',
                         'model': partial(biased_svm.biased_SVM_weight_selection, P_train_, U_train_)},
                        # {'name' : 'bagging-svm',
                        #  'model': biased_svm.biased_SVM_grid_search(P_train_, U_train_)}
                    ]

                    # eval models
                    # TODO multiprocessing; breaks on macOS but not on Linux
                    if PARALLEL:
                        with multi.Pool(min(multi.cpu_count(), len(iteration))) as p:
                            iter_stats = list(p.map(partial(model_eval_record, X_dev_, y_test, U_train_), iteration,
                                                    chunksize=1
                                                    ))
                    else:
                        iter_stats = list(map(partial(model_eval_record, X_dev_, y_test, U_train_), iteration))

                    # finalize records: remove model, add n-gram stats, update best
                    for m in iter_stats:
                        m['n-grams'] = pp
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

    # save results to disk

    with open(file_path("./pickles/model_eval{}.pickle".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
              "wb") as f:
        print('saving model stats to disk\n')
        pickle.dump(results, f)

    # ----------------------------------------------------------------
    # check how much of U (abstracts) is supposed to be positive
    # ----------------------------------------------------------------

    best_model = results['best']['model']
    selector = results['best']['selector']
    vectorizer = results['best']['vectorizer']

    print("\nAmount of unlabelled training set classified as positive:")
    if selector:
        transformedU = (selector.transform(vectorizer.transform(U_train)))
    else:
        transformedU = (vectorizer.transform(U_train))
    y_predicted_U = best_model.predict(transformedU)
    print(np.sum(y_predicted_U), "/", num_rows(y_predicted_U),
          "(", np.sum(y_predicted_U) / num_rows(y_predicted_U), ")")

    return results['best']


def prepareTrainTest(trainData, testData, trainLabels, rules=True, wordgram_range=None, feature_select=None,
                     chargram_range=None, genia_opts=None, min_df_char=0.001, min_df_word=0.001, max_df=1.0):
    """prepare training and test vectors and vectorizer for validating classifiers
    """

    print("Fitting vectorizer, preparing training and test data")

    vectorizer = transformers.vectorizer(chargrams=chargram_range, min_df_char=min_df_char, wordgrams=wordgram_range,
                                         min_df_word=min_df_word, genia_opts=genia_opts, rules=rules)

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

    # print("\n{}:\tacc: {}, classification report:\n{}".format(name, acc, clsr))

    return {'name': name, 'p': p, 'r': r, 'f1': f1, 'acc': acc, 'clsr': clsr, 'model': model, 'U_ratio': pos_ratio}


def print_results(results):
    best = results['best']

    print("Best:")
    print(best['name'], best['n-grams'],
          "\tstats: p={}\tr={}\tf1={}\tacc={}\t".format(best['p'], best['r'], best['f1'], best['acc']), "\n")
    print("amount of U labelled as relevant:", best['U_ratio'])

    print("All stats:")
    for i in results['all']:
        print_reports(i)
    return


def print_reports(i):
    print(i[0]['n-grams'], i[0]['fs'])
    for m in i:
        print("\n{}:\tacc: {}, relevant ratio in U: {}, classification report:\n{}".format(
                m['name'], m['acc'], m['U_ratio'], m['clsr']))
    return


def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)
