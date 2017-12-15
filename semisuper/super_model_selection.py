import datetime
import multiprocessing as multi
import os
import pickle
import time
from copy import deepcopy
from functools import partial
from itertools import product

from scipy.stats import randint as sp_randint, uniform

import numpy as np
from scipy.sparse import vstack
from sklearn.metrics import classification_report, precision_recall_fscore_support, \
    accuracy_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from semisuper import transformers
from semisuper.helpers import num_rows, densify

PARALLEL = False  # TODO multiprocessing works on Linux when there aren't too many features, but not on macOS
RAND_INT_MAX = 1000


def best_model_cross_val(X, y, fold=10):
    """determine best model, cross validate and return pipeline trained on all data"""

    print("\nFinding best model\n")

    best = get_best_model(X, y)

    print("\nCross-validation\n")

    kf = KFold(n_splits=fold, shuffle=True)
    splits = kf.split(X, y)

    if PARALLEL:
        with multi.Pool(fold) as p:
            stats = list(p.map(partial(eval_fold, best, X, y), enumerate(splits), chunksize=1))
    else:
        stats = list(map(partial(eval_fold, best, X, y), enumerate(splits)))

    mean_stats = np.mean(stats, 0)
    print("Cross-validation average: p {}, r {}, f1 {}, acc {}".format(
            mean_stats[0], mean_stats[1], mean_stats[2], mean_stats[3]))

    print("Retraining model on full data")

    best.fit(X, y)

    print("Returning final model")

    return best


# helper
def eval_fold(model, X, y, i_splits):
    """helper function for running cross validation in parallel"""

    i, split = i_splits
    X_train, X_test = X[split[0]], X[split[1]]
    y_train, y_test = y[split[0]], y[split[1]]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    pr, r, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("Fold no.", i, "acc", acc, "classification report:\n", classification_report(y_test, y_pred))
    return [pr, r, f1, acc]


def names_estimators_params():
    l = [
        {"name"  : "LinearSVC",
         "model" : LinearSVC(),
         "params": {'C'   : uniform(0, 1),
                    'loss': ['hinge', 'squared_hinge']
                    }
         },
        {"name"  : "MultinomialNB",
         "model" : MultinomialNB(),
         "params": {'alpha'    : uniform(0, 1),
                    'fit_prior': [True],
                    }
         },
        {"name"  : "LogisticRegression",
         "model" : LogisticRegression(),
         "params": {'C'           : sp_randint(1, RAND_INT_MAX),
                    'solver'      : ['newton-cg', 'lbfgs', 'liblinear'],  # 'sag', 'saga'
                    'class_weight': ['balanced']
                    }
         },
        {"name"  : "SGDClassifier",
         "model" : SGDClassifier(),
         "params": {
             'loss'         : ['hinge', 'log', 'modified_huber', 'squared_hinge'],
             'class_weight' : ['balanced'],
             'penalty'      : ['l2', 'l1', 'elasticnet'],
             'learning_rate': ['optimal', 'invscaling'],
             'eta0'         : uniform(0.01, 0.00001)
         }
         },
        {"name"  : "SVM_SVC",
         "model" : SVC(),
         "params": {'C'           : sp_randint(1, RAND_INT_MAX),
                    'kernel'      : ['linear', 'poly', 'rbf', 'sigmoid'],
                    'class_weight': ['balanced'],
                    'probability' : [True]
                    }
         },
        {"name"  : "Lasso",
         "model" : Lasso(),
         "params": {'alpha'        : uniform(0, 1),
                    'fit_intercept': [True],
                    'normalize'    : [True, False],
                    'max_iter'     : sp_randint(1, RAND_INT_MAX)
                    }
         },
        {"name"  : "ElasticNet",
         "model" : ElasticNet(),
         "params": {'alpha'   : uniform(0, 1),
                    'l1_ratio': uniform(0, 1)
                    }
         },
        {"name"  : "MLPClassifier",
         "model" : MLPClassifier(),
         "params": {'activation'   : ['identity', 'logistic', 'tanh', 'relu'],
                    'solver'       : ['lbfgs', 'sgd', 'adam'],
                    'learning_rate': ['constant', 'invscaling', 'adaptive'],
                    'max_iter'     : [100000]
                    }
         },
        {"name"  : "DecisionTreeClassifier",
         "model" : DecisionTreeClassifier(),
         "params": {"criterion"   : ["gini", "entropy"],
                    "splitter"    : ["best", "random"],
                    'max_depth'   : sp_randint(1, 1000),
                    'class_weight': ['balanced']
                    }
         },
        {"name"  : "RandomForestClassifier",
         "model" : RandomForestClassifier(),
         "params": {'n_estimators': sp_randint(1, RAND_INT_MAX),
                    "criterion"   : ["gini", "entropy"],
                    'max_depth'   : sp_randint(1, RAND_INT_MAX),
                    'class_weight': ['balanced']
                    }
         },
        {"name"  : "KNeighbors",
         "model" : KNeighborsClassifier(),
         "params": {'n_neighbors' : sp_randint(1, 40),
                    'weights'     : ['uniform', 'distance'],
                    'algorithm'   : ['auto'],
                    'leaf_size'   : sp_randint(1, RAND_INT_MAX),
                    'class_weight': ['balanced']
                    }
         },
    ]

    return l


def get_best_model(X_train, y_train, X_test=None, y_test=None):
    """Evaluate parameter combinations, save results and return object with stats of all models"""

    print("\nEvaluating parameter ranges for preprocessor and classifiers")

    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

    X_eval, X_dev, y_eval, y_dev = train_test_split(X_test, y_test, test_size=0.5)

    results = {'best': {'f1': -1, 'acc': -1}, 'all': []}

    preproc_params = {
        'df_min'        : [0.001],
        'df_max'        : [1.0],
        'rules'         : [True],  # [True, False],
        'lemmatize'     : [False],
        'wordgram_range': [None, (1, 2), (1, 3), (1, 4)],
        'chargram_range': [None, (2, 4), (2, 5), (2, 6)],
        'feature_select': [

            # partial(basic_pipeline.percentile_selector, 'chi2', 30),
            partial(transformers.percentile_selector, 'chi2', 25),
            # partial(basic_pipeline.percentile_selector, 'chi2', 20),
            # partial(basic_pipeline.percentile_selector, 'f', 30),
            # partial(basic_pipeline.percentile_selector, 'f', 25),
            # partial(basic_pipeline.percentile_selector, 'f', 20),
            # partial(basic_pipeline.percentile_selector, 'mutual_info', 30), # mutual information: worse than rest
            # partial(basic_pipeline.percentile_selector, 'mutual_info', 25),
            # partial(basic_pipeline.percentile_selector, 'mutual_info', 20),
            partial(basic_pipeline.factorization, 'TruncatedSVD', 1000),
            # partial(basic_pipeline.factorization, 'TruncatedSVD', 2000), # 10% worse than chi2, slow, SVM iter >100
            # partial(basic_pipeline.factorization, 'TruncatedSVD', 3000),
        ]
    }

    estimators = names_estimators_params()

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

                    if PARALLEL:
                        with multi.Pool(min(multi.cpu_count(), len(estimators))) as p:
                            iter_stats = list(p.map(partial(model_eval_record,
                                                            X_train_, y_train, X_dev_, y_dev),
                                                    estimators, chunksize=1))
                    else:
                        iter_stats = list(map(partial(model_eval_record,
                                                      X_train_, y_train, X_dev_, y_dev),
                                              estimators))

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


def model_eval_record(X_train, y_train, X_test, y_test, model_params):
    """helper function for finding best model in parallel: evaluate model and return stat object. """

    random_search = RandomizedSearchCV(model_params['model'],
                                       param_distributions=model_params['params'],
                                       n_iter=20,
                                       n_jobs=-1,
                                       pre_dispatch='n_jobs',
                                       cv=10,
                                       scoring='f1_macro',
                                       verbose=0)

    random_search.fit(X_train, y_train)
    model = random_search.best_estimator_
    params = random_search.best_params_
    y_pred = model.predict(X_test)

    name = model_params['name']
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    clsr = classification_report(y_test, y_pred)

    print("\n{} with params{}:\nacc: {}, classification report:\n{}".format(name, params, acc, clsr))
    return {'name' : name, 'p': p, 'r': r, 'f1': f1, 'acc': acc, 'clsr': clsr,
            'model': model, 'params': params}


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
