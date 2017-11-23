import datetime
import os
import pickle

import numpy as np
import semisuper.basic_pipeline as pipeline
import semisuper.pu_biased_svm as biased_svm
import semisuper.pu_two_step as two_step
from semisuper.helpers import num_rows, unsparsify
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
import multiprocessing as multi
from functools import partial
import time
from copy import deepcopy


def getBestModel(P_train, U_train, X_test, y_test):
    """Evaluate parameter combinations, save results and return pipeline with best model"""

    print("\nEvaluating parameter ranges for preprocessor and classifiers")

    # splitting test set (should have true labels) in test and dev set
    X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.3)

    X_train = np.concatenate((P_train, U_train), 0)
    y_train = np.concatenate((np.ones(num_rows(P_train)), np.zeros(num_rows(U_train))))

    results = {'best': {'f1': -1, 'acc': -1}, 'all': []}

    preproc_params = {
        'df_min'        : [1],
        'df_max'        : [1.0],
        'rules'         : [True],
        'lemmatize'     : [False],
        'wordgram_range': [None, (1, 2), (1, 3), (1, 4)],
        'chargram_range': [(2, 4), (2, 5), (2, 6)]
    }

    for wordgram in preproc_params['wordgram_range']:
        for chargram in preproc_params['chargram_range']:
            for r in preproc_params['rules']:
                for l in preproc_params['lemmatize']:

                    print("\n----------------------------------------------------------------",
                          "\nwords:", wordgram, "chars:", chargram,
                          "\n----------------------------------------------------------------\n")

                    start_time = time.time()

                    X_train_, X_dev_, vectorizer, selector = prepareTrainTest(trainData=X_train, trainLabels=y_train,
                                                                              testData=X_dev,
                                                                              wordgram_range=wordgram,
                                                                              chargram_range=chargram,
                                                                              rules=r, lemmatize=l)
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
                        {'name': 'i-em', 'model': two_step.i_EM(P_train_, U_train_)},
                        {'name' : 's-em spy=0.1',
                         'model': two_step.s_EM(P_train_, U_train_, spy_ratio=0.1, noise_lvl=0.1)},
                        {'name' : 's-em spy=0.2',
                         'model': two_step.s_EM(P_train_, U_train_, spy_ratio=0.2, noise_lvl=0.2)},
                        {'name': 'roc-svm', 'model': two_step.roc_SVM(P_train_, U_train_)},
                        {'name': 'cr_svm noise=0.1', 'model': two_step.cr_SVM(P_train_, U_train_, noise_lvl=0.1)},
                        {'name': 'cr_svm noise=0.2', 'model': two_step.cr_SVM(P_train_, U_train_, noise_lvl=0.2)},
                        {'name': 'roc_em', 'model': two_step.roc_EM(P_train_, U_train_)},
                        {'name' : 'spy_svm spy=0.1',
                         'model': two_step.spy_SVM(P_train_, U_train_, spy_ratio=0.1, noise_lvl=0.1)},
                        {'name' : 'spy_svm spy=0.2',
                         'model': two_step.spy_SVM(P_train_, U_train_, spy_ratio=0.2, noise_lvl=0.2)},
                        {'name' : 'biased-svm',
                         'model': biased_svm.biased_SVM_weight_selection(P_train_, U_train_)},
                        {'name' : 'bagging-svm',
                         'model': biased_svm.biased_SVM_grid_search(P_train_, U_train_)}
                    ]

                    # eval models
                    with multi.Pool(min(multi.cpu_count(), len(iteration))) as p:
                        iter_stats = p.map(partial(model_eval_record, X_dev_, y_dev), iteration)

                    # finalize records: remove memory-heavy model, add n-gram stats, update best
                    for m in iter_stats:
                        m['n-grams'] = pp
                        if m['acc'] > results['best']['acc']:
                            results['best'] = deepcopy(m)
                            results['best']['vectorizer'] = vectorizer
                            results['best']['selector'] = selector
                        m.pop('model', None)

                    print("Evaluated words:", wordgram, "chars:", chargram,
                          "in %s seconds\n" % (time.time() - start_time))
                    results['all'].append(iter_stats)

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
            transformedTestData = unsparsify(vectorizer.transform(X_train))

        classModel = best_model.fit(transformedData, [1] * num_rows(P_train) + [0] * num_rows(U_train))

        # ===============================================================
        # PERFORMANCE OF MODEL ON TEST DATA
        # ===============================================================

        y_predicted_test = classModel.predict(transformedTestData)
        print(classification_report(y_test, y_predicted_test))

    return results['best']


def prepareTrainTest(trainData, testData, trainLabels, featureSelect=True, min_df=1, max_df=1.0,
                     wordgram_range=None, chargram_range=None, rules=True, lemmatize=True):
    """prepare training and test vectors and vectorizer for validating classifiers"""

    print("Fitting vectorizer and preparing training and test data")

    vectorizer = pipeline.vectorizer(words=True if wordgram_range else False, wordgram_range=wordgram_range,
                                     chars=True if chargram_range else False, chargram_range=chargram_range,
                                     rules=rules, lemmatize=lemmatize, min_df=min_df, max_df=max_df)

    transformedTrainData = vectorizer.fit_transform(trainData)
    transformedTestData = vectorizer.transform(testData)

    print("No. of features:", transformedTrainData.shape[1])

    selector = None
    if featureSelect:
        selector = SelectPercentile(chi2, 20)
        selector.fit(transformedTrainData, trainLabels)
        transformedTrainData = selector.transform(transformedTrainData)
        transformedTestData = selector.transform(transformedTestData)

        print("No. of features after reduction:", transformedTrainData.shape[1], "\n")

    return transformedTrainData, transformedTestData, vectorizer, selector


def model_eval_record(X, y, m):
    model = m['model']
    name = m['name']

    y_pred = model.predict(X)

    p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average='macro')
    acc = accuracy_score(y, y_pred)
    clsr = classification_report(y, y_pred)

    print('Classification report for', name, '\n', clsr)

    return {'name': name, 'p': p, 'r': r, 'f1': f1, 'acc': acc, 'clsr': clsr, 'model': model}


def print_results(results):
    best = results['best']

    print("Best:")
    print(best['name'], best['n-grams'],
          "\tstats: p={}\tr={}\tf1={}\tacc={}\t".format(best['p'], best['r'], best['f1'], best['acc']), "\n")

    print("All stats:")
    for i in results['all']:
        print(i[0]['n-grams'])
        for m in i:
            print("\t", m['name'], "\n\t\t",
                  "stats: p={}\tr={}\tf1={}\tacc={}\t".format(best['p'], best['r'], best['f1'], best['acc']))
    return


def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)
