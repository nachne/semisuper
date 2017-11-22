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


def getBestModel(P, U, X_test=None, y_test=None):
    """evaluate parameter combinations, save results and return pipeline with best model"""

    print("\nEvaluating parameters for preprocessor and classifiers\n")

    # TODO not really applicable for PU!
    P_train, P_dev = train_test_split(P)
    U_train, U_dev = train_test_split(U)

    X_train = np.concatenate((P_train, U_train), 0)
    y_train = np.concatenate((np.ones(num_rows(P_train)), np.zeros(num_rows(U_train))))

    X_dev = np.concatenate((P_dev, U_dev), 0)
    y_dev = np.concatenate((np.ones(num_rows(P_dev)), np.zeros(num_rows(U_dev))))

    results = {'best': {'f1': -1}, 'all': []}

    preproc_params = {
        'df_min': [1],
        'df_max': [1.0],
        'rules': [True],
        'lemmatize': [False],
        'wordgram_range': [None, (1, 2), (1, 3), (1, 4)],
        'chargram_range': [(2, 4), (2, 5), (2, 6)]
    }

    for wordgram in preproc_params['wordgram_range']:
        for chargram in preproc_params['chargram_range']:
            for r in preproc_params['rules']:
                for l in preproc_params['lemmatize']:

                    print("\n----------------------------------------------------------------",
                          "\n----------------------------------------------------------------",
                          "\nwords:", wordgram, "chars:", chargram, "\n",
                          "\n----------------------------------------------------------------",
                          "\n----------------------------------------------------------------\n")

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

                    iteration = {
                        'models': [
                            {'name': 'i-em', 'model': two_step.i_EM(P_train_, U_train_)},
                            {'name': 's-em spy=0.1',
                             'model': two_step.s_EM(P_train_, U_train_, spy_ratio=0.1, noise_lvl=0.1)},
                            {'name': 's-em spy=0.2',
                             'model': two_step.s_EM(P_train_, U_train_, spy_ratio=0.2, noise_lvl=0.2)},
                            {'name': 'roc-svm', 'model': two_step.roc_SVM(P_train_, U_train_)},
                            {'name': 'cr_svm noise=0.1', 'model': two_step.cr_SVM(P_train_, U_train_, noise_lvl=0.1)},
                            {'name': 'cr_svm noise=0.2', 'model': two_step.cr_SVM(P_train_, U_train_, noise_lvl=0.2)},
                            {'name': 'roc_em', 'model': two_step.roc_EM(P_train_, U_train_)},
                            {'name': 'spy_svm spy=0.1',
                             'model': two_step.spy_SVM(P_train_, U_train_, spy_ratio=0.1, noise_lvl=0.1)},
                            {'name': 'spy_svm spy=0.2',
                             'model': two_step.spy_SVM(P_train_, U_train_, spy_ratio=0.2, noise_lvl=0.2)},
                            {'name': 'biased-svm',
                             'model': biased_svm.biased_SVM_weight_selection(P_train_, U_train_)}
                        ]
                    }
                    # TODO parallel
                    for m in iteration['models']:
                        y_pred = m['model'].predict(X_dev_)

                        m['p'], m['r'], m['f1'], _ = precision_recall_fscore_support(y_dev, y_pred, average='macro')
                        m['clsr'] = classification_report(y_dev, y_pred)

                        m['preprocessing'] = {'word': wordgram, 'char': chargram,
                                              'vectorizer': vectorizer, 'selector': selector}

                        m.pop('model')

                        if m['f1'] > results['best']['f1']:
                            results['best'] = m

                        results['all'].append(iteration)

    with open(file_path("./pickles/model_eval{}.pickle".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
              "wb") as f:
        print('saving all models to disk')
        pickle.dump(results, f)

        print(results)

    best_model = results['best']

    if X_test:
        if selector:
            transformedTestData = unsparsify(selector.transform(vectorizer.transform(X_test)))
        else:
            transformedTestData = unsparsify(vectorizer.transform(X_test))

        y_pred_test = best_model.predict(transformedTestData)

        p, r, f, s = precision_recall_fscore_support(y_test, y_pred_test, average='macro')
        acc = accuracy_score(y_test, y_pred_test)

        print('\tTEST:\t\t\t', p, r, f, acc)
        print(results['best'])

        # ===============================================================
        # CREATE MODEL BASED ON BEST
        # ===============================================================
        print('Fitting best model on complete data')

        vectorizer = best_model['preprocessing']['vectorizer']
        selector = best_model['preprocessing']['selector']
        model = best_model['model']

        if selector:
            transformedData = unsparsify(selector.fit_transform(vectorizer.fit_transform(
                np.concatenate((P_train, U_train), 0)),
                [1] * num_rows(P_train) + [0] * num_rows(U_train)))
            transformedTestData = unsparsify(selector.transform(vectorizer.transform(X_test), y_test))
        else:
            transformedData = unsparsify(vectorizer.fit_transform(np.concatenate((P_train, U_train), 0)))
            transformedTestData = unsparsify(vectorizer.transform(X_train))

        classModel = model.fit(transformedData, [1] * num_rows(P_train) + [0] * num_rows(U_train))

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
        transformedTrainData = selector.transform(transformedTrainData, trainLabels)
        transformedTestData = selector.transform(transformedTestData)

        print("No. of features after reduction:", transformedTrainData.shape[1], "\n")

    return transformedTrainData, transformedTestData, vectorizer, selector


def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)
