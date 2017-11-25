import datetime
import multiprocessing as multi
import os
import pickle
import time
from copy import deepcopy
from functools import partial

import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split

import semisuper.basic_pipeline as basic_pipeline
import semisuper.pu_biased_svm as biased_svm
import semisuper.pu_two_step as two_step
from semisuper.helpers import num_rows, unsparsify
from semisuper.loaders import sentences_civic_abstracts


def getBestModel(P_train, U_train, X_test, y_test):
    """Evaluate parameter combinations, save results and return pipeline with best model"""

    print("\nEvaluating parameter ranges for preprocessor and classifiers")

    # splitting test set (should have true labels) in test and dev set
    X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.5)

    X_train = np.concatenate((P_train, U_train), 0)
    y_train = np.concatenate((np.ones(num_rows(P_train)), np.zeros(num_rows(U_train))))

    results = {'best': {'f1': -1, 'acc': -1}, 'all': []}

    preproc_params = {
        'df_min'        : [1],
        'df_max'        : [1.0],
        'rules'         : [True],
        'lemmatize'     : [False],
        'wordgram_range': [None, (1, 2), (1, 3), (1, 4)],
        'chargram_range': [None, (2, 4), (2, 5), (2, 6)]
    }

    for wordgram in preproc_params['wordgram_range']:
        for chargram in preproc_params['chargram_range']:
            for r in preproc_params['rules']:
                for l in preproc_params['lemmatize']:

                    if wordgram == None and chargram == None:
                        break

                    print("\n----------------------------------------------------------------",
                          "\nwords:", wordgram, "chars:", chargram,
                          "\n----------------------------------------------------------------\n")

                    start_time = time.time()

                    X_train_, X_dev_, vectorizer, selector = prepareTrainTest(trainData=X_train, testData=X_dev,
                                                                              trainLabels=y_train, rules=r,
                                                                              wordgram_range=wordgram,
                                                                              featureSelect=False,
                                                                              chargram_range=chargram, lemmatize=l)
                    if selector:
                        P_train_ = selector.transform(vectorizer.transform(P_train))
                        U_train_ = selector.transform(vectorizer.transform(U_train))
                    else:
                        P_train_ = vectorizer.transform(P_train)
                        U_train_ = vectorizer.transform(U_train)
                    P_train_ = unsparsify(P_train_)
                    U_train_ = unsparsify(U_train_)
                    X_train_ = unsparsify(X_train_)
                    X_dev_ = unsparsify(X_dev_)

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
                    with multi.Pool(min(multi.cpu_count(), len(iteration))) as p:
                        iter_stats = list(p.map(partial(model_eval_record, X_dev_, y_dev), iteration, chunksize=1))

                    # finalize records: remove model, add n-gram stats, update best
                    for m in iter_stats:
                        m['n-grams'] = pp
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

    with open(file_path("./pickles/model_eval{}.pickle".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
              "wb") as f:
        print('saving model stats to disk\n')
        pickle.dump(results, f)

    if X_test:
        best_model = results['best']['model']
        selector = results['best']['selector']
        vectorizer = results['best']['vectorizer']

        if selector:
            transformedTestData = unsparsify(selector.transform(vectorizer.transform(X_test)))
        else:
            transformedTestData = unsparsify(vectorizer.transform(X_test))

        y_pred_test = best_model.predict(transformedTestData)

        p, r, f, s = precision_recall_fscore_support(y_test, y_pred_test, average='macro')
        acc = accuracy_score(y_test, y_pred_test)

        print("TEST:", results['best']['name'], results['best']['n-grams'], "\n",
              'p={}\tr={}\tf1={}\tacc={}'.format(p, r, f, acc))

        # ===============================================================
        # CREATE MODEL BASED ON BEST
        # ===============================================================
        print('\nFitting best model on complete data')

        if selector:
            transformedData = unsparsify(selector.fit_transform(vectorizer.fit_transform(
                    np.concatenate((P_train, U_train), 0)),
                    [1] * num_rows(P_train) + [0] * num_rows(U_train)))
            transformedTestData = unsparsify(selector.transform(vectorizer.transform(X_test)))
        else:
            transformedData = unsparsify(vectorizer.fit_transform(np.concatenate((P_train, U_train), 0)))
            transformedTestData = unsparsify(vectorizer.transform(X_test))

        classModel = best_model.fit(transformedData, [1] * num_rows(P_train) + [0] * num_rows(U_train))

        # ===============================================================
        # PERFORMANCE OF MODEL ON TEST DATA
        # ===============================================================

        y_predicted_test = classModel.predict(transformedTestData)
        print(classification_report(y_test, y_predicted_test))

    return results['best']


def prepareTrainTest(trainData, testData, trainLabels, rules=True, wordgram_range=None, featureSelect=True,
                     chargram_range=None, lemmatize=True, min_df_char=100, min_df_word=50, max_df=1.0):
    """prepare training and test vectors and vectorizer for validating classifiers
    :param min_df_char:
    """

    print("Fitting vectorizer, preparing training and test data")

    vectorizer = basic_pipeline.vectorizer(words=True if wordgram_range else False, wordgram_range=wordgram_range,
                                           chars=True if chargram_range else False, chargram_range=chargram_range,
                                           rules=rules, lemmatize=lemmatize, min_df_word=min_df_word,
                                           min_df_char=min_df_char)

    transformedTrainData = vectorizer.fit_transform(trainData)
    transformedTestData = vectorizer.transform(testData)

    print("No. of features:", transformedTrainData.shape[1])

    selector = None
    if featureSelect:
        selector = basic_pipeline.selector()
        selector.fit(transformedTrainData, trainLabels)
        transformedTrainData = selector.transform(transformedTrainData)
        transformedTestData = selector.transform(transformedTestData)

        print("No. of features after reduction:", transformedTrainData.shape[1], "\n")
    print()
    return transformedTrainData, transformedTestData, vectorizer, selector


def model_eval_record(X, y, m):
    model = m['model']()
    name = m['name']

    y_pred = model.predict(X)

    p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average='macro')
    acc = accuracy_score(y, y_pred)
    clsr = classification_report(y, y_pred)

    print("\n{}:\tacc: {}, classification report:\n{}".format(name, acc, clsr))

    return {'name': name, 'p': p, 'r': r, 'f1': f1, 'acc': acc, 'clsr': clsr, 'model': model}


def print_results(results):
    best = results['best']

    print("Best:")
    print(best['name'], best['n-grams'],
          "\tstats: p={}\tr={}\tf1={}\tacc={}\t".format(best['p'], best['r'], best['f1'], best['acc']), "\n")

    print("All stats:")
    for i in results['all']:
        print_stats(i)
    return


def print_stats(i):
    print(i[0]['n-grams'])
    for m in i:
        print("\t", m['name'], "\n\t\t",
              "stats: p={}\tr={}\tf1={}\tacc={}\t".format(m['p'], m['r'], m['f1'], m['acc']))
        print(m['clsr'])
    return


def print_reports(i):
    print(i[0]['n-grams'])
    for m in i:
        print("\n{}:\tacc: {}, classification report:\n{}".format(m['name'], m['acc'], m['clsr']))
    return

def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)
