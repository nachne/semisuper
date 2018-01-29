from __future__ import absolute_import, division, print_function

import multiprocessing as multi
import time
from copy import deepcopy
from functools import partial
from itertools import product

from scipy.stats import randint as sp_randint, uniform

import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from semisuper import transformers
from semisuper.helpers import densify

PARALLEL = True
RAND_INT_MAX = 1000
RANDOM_SEED = 4242


# ----------------------------------------------------------------
# Estimators and parameters to evaluate
# ----------------------------------------------------------------

def estimator_list():
    l = [
        {"name"  : "LinearSVC",
         "model" : LinearSVC(),
         "params": {'C'   : uniform(0.5, 0.5),
                    'loss': ['hinge', 'squared_hinge']
                    }
         },
        {"name"  : "LogisticRegression",
         "model" : LogisticRegression(),
         "params": {'C'           : sp_randint(1, RAND_INT_MAX),
                    'solver'      : ['lbfgs'],  # ['newton-cg', 'lbfgs', 'liblinear'],  # 'sag', 'saga'
                    'class_weight': ['balanced']
                    }
         },
        {"name"  : "SGDClassifier",
         "model" : SGDClassifier(),
         "params": {'loss'         : ['hinge', 'log', 'modified_huber', 'squared_hinge'],
                    'class_weight' : ['balanced'],
                    'penalty'      : ['l2', 'l1', 'elasticnet'],
                    'learning_rate': ['optimal', 'invscaling'],
                    'max_iter'     : [1000],  # for sklearn >= 0.19, not 0.18
                    'tol'          : [1e-3],  # for sklearn >= 0.19, not 0.18
                    'eta0'         : uniform(0.01, 0.00001)
                    }
         },

        ## SVC: slow!
        # {"name"  : "SVM_SVC",
        #  "model" : SVC(),
        #  "params": {'C'           : sp_randint(1, RAND_INT_MAX),
        #             'kernel'      : ['poly', 'rbf', 'sigmoid'],
        #             'class_weight': ['balanced'],
        #             'probability' : [False]
        #             }
        #  },

        ## MNB: bad performance (< 80% avg, < 70% recall)
        # {"name"  : "MultinomialNB",
        #  "model" : MultinomialNB(),
        #  "params": {'alpha'    : uniform(0, 1),
        #             'fit_prior': [True],
        #             }
        #  },

        ## Lasso, ElascticNet: mix of continuous and discrete labels
        # {"name"  : "Lasso",
        #  "model" : Lasso(),
        #  "params": {'alpha'        : uniform(0, 1),
        #             'fit_intercept': [True],
        #             'normalize'    : [True, False],
        #             'max_iter'     : sp_randint(1, RAND_INT_MAX)
        #             }
        #  },
        # {"name"  : "ElasticNet",
        #  "model" : ElasticNet(),
        #  "params": {'alpha'   : uniform(0, 1),
        #             'l1_ratio': uniform(0, 1)
        #             }
        #  },
        ## DecisionTree: bad performance
        # {"name"  : "DecisionTreeClassifier",
        #  "model" : DecisionTreeClassifier(),
        #  "params": {"criterion"   : ["gini", "entropy"],
        #             "splitter"    : ["best", "random"],
        #             'max_depth'   : sp_randint(1, 1000),
        #             'class_weight': ['balanced']
        #             }
        #  },
        ## {"name"  : "RandomForestClassifier",
        #  "model" : RandomForestClassifier(),
        #  "params": {'n_estimators': sp_randint(1, RAND_INT_MAX),
        #             "criterion"   : ["gini", "entropy"],
        #             'max_depth'   : sp_randint(1, RAND_INT_MAX),
        #             'class_weight': ['balanced']
        #             }
        #  },
        ## {"name"  : "KNeighbors",
        #  "model" : KNeighborsClassifier(),
        #  "params": {'n_neighbors' : sp_randint(1, 40),
        #             'weights'     : ['uniform', 'distance'],
        #             'algorithm'   : ['auto'],
        #             'leaf_size'   : sp_randint(1, RAND_INT_MAX)
        #             }
        #  },
        ## MLP: crashes
        # {"name"  : "MLPClassifier",
        #  "model" : MLPClassifier(),
        #  "params": {'activation'   : ['identity', 'logistic', 'tanh', 'relu'],
        #             'solver'       : ['lbfgs', 'sgd', 'adam'],
        #             'learning_rate': ['constant', 'invscaling', 'adaptive'],
        #             'max_iter'     : [1000],
        #             }
        #  },
    ]

    return l[:1]


def preproc_param_dict():
    d = {
        'df_min'        : [0.001],
        'df_max'        : [1.0],
        'rules'         : [True],  # [True, False],
        'genia_opts'    : [None, {"pos": False, "ner": False}],
        # [None, {"pos": False, "ner": False}, {"pos": True, "ner": False}, {"pos": False, "ner": True},
        # {"pos": True, "ner": True}],
        'wordgram_range': [(1, 4)],  # [(1, 3), (1, 4)], # [None, (1, 2), (1, 3), (1, 4)],
        'chargram_range': [(2, 6)],  # [(2, 5), (2, 6)], # [None, (2, 4), (2, 5), (2, 6)],
        'feature_select': [
            # transformers.IdentitySelector,
            # partial(transformers.percentile_selector, 'chi2', 30),
            partial(transformers.percentile_selector, 'chi2', 25),
            # partial(transformers.percentile_selector, 'chi2', 20),
            # partial(transformers.percentile_selector, 'f', 30),
            # partial(transformers.percentile_selector, 'f', 25),
            # partial(transformers.percentile_selector, 'f', 20),
            # partial(transformers.percentile_selector, 'mutual_info', 30), # mutual information: worse than rest
            # partial(transformers.percentile_selector, 'mutual_info', 25),
            # partial(transformers.percentile_selector, 'mutual_info', 20),
            # partial(transformers.factorization, 'LatentDirichletAllocation', 100),
            # partial(transformers.factorization, 'TruncatedSVD', 100),
            # partial(transformers.factorization, 'TruncatedSVD', 1000),
            # partial(transformers.factorization, 'TruncatedSVD', 2000), # 10% worse than chi2, slow, SVM iter >100
            # partial(transformers.factorization, 'TruncatedSVD', 3000),
            # partial(transformers.select_from_l1_svc, 1.0, 1e-3),
            # partial(transformers.select_from_l1_svc, 0.5, 1e-3),
            # partial(transformers.select_from_l1_svc, 0.1, 1e-3),
        ]
    }
    return d


# ----------------------------------------------------------------
# Cross validation
# ----------------------------------------------------------------

def best_model_cross_val(X, y, fold=10):
    """determine best model, cross validate and return pipeline trained on all data"""

    print("\nFinding best model\n")

    best = get_best_model(X, y)

    print("\nCross-validation\n")

    kf = KFold(n_splits=fold, shuffle=True)
    splits = kf.split(X, y)

    # TODO: parallel fix
    # if PARALLEL:
    #     with multi.Pool(fold) as p:
    #         stats = list(p.map(partial(eval_fold, best, X, y), enumerate(splits), chunksize=1))
    # else:
    #     stats = list(map(partial(eval_fold, best, X, y), enumerate(splits)))

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


# ----------------------------------------------------------------
# Model selection
# ----------------------------------------------------------------

def get_best_model(X_train, y_train, X_test=None, y_test=None):
    """Evaluate parameter combinations, save results and return object with stats of all models"""

    print("\nEvaluating parameter ranges for preprocessor and classifiers")

    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_SEED)

    results = {'best': {'f1': -1, 'acc': -1}, 'all': []}

    preproc_params = preproc_param_dict()
    estimators = estimator_list()

    for wordgram, chargram in product(preproc_params['wordgram_range'], preproc_params['chargram_range']):
        for r, genia_opts in product(preproc_params['rules'], preproc_params['genia_opts']):
            for df_min, df_max in product(preproc_params['df_min'], preproc_params['df_max']):
                for fs in preproc_params['feature_select']:

                    if wordgram is None and chargram is None:
                        break

                    print("\n----------------------------------------------------------------",
                          "\nwords:", wordgram, "chars:", chargram, "feature selection:", fs,
                          "df_min, df_max:", df_min, df_max, "rules, genia_opts:", r, genia_opts,
                          "\n----------------------------------------------------------------\n")

                    start_time = time.time()

                    X_train_, X_test_, vectorizer, selector = prepare_train_test(trainData=X_train, testData=X_test,
                                                                                 trainLabels=y_train, rules=r,
                                                                                 wordgram_range=wordgram,
                                                                                 feature_select=fs,
                                                                                 chargram_range=chargram,
                                                                                 genia_opts=genia_opts,
                                                                                 min_df_char=df_min,
                                                                                 min_df_word=df_min, max_df=df_max)

                    # fit models
                    with multi.Pool(multi.cpu_count()) as p:
                        iter_stats = list(p.map(partial(model_eval_record, X_train_, y_train, X_test_, y_test),
                                                estimators))

                    # finalize records: remove model, add n-gram stats, update best
                    for m in iter_stats:
                        m['n-grams'] = {'word': wordgram, 'char': chargram},
                        m['rules, genia_opts'] = (r, genia_opts)
                        m['df_min, df_max'] = (df_min, df_max)
                        m['fs'] = fs()
                        if m['acc'] > results['best']['acc']:
                            results['best'] = deepcopy(m)
                            results['best']['vectorizer'] = vectorizer
                            results['best']['selector'] = selector
                        m.pop('model', None)

                    results['all'].append(iter_stats)

                    print("Evaluated words:", wordgram, "chars:", chargram, "rules:", r, "genia:", genia_opts,
                          "feature selection:", fs, "min_df:", df_min,
                          "in %s seconds\n" % (time.time() - start_time))

    # print_results(results)

    return Pipeline([('vectorizer', results['best']['vectorizer']),
                     ('selector', results['best']['selector']),
                     ('clf', results['best']['model'])])


def model_eval_record(X_train, y_train, X_test, y_test, model_params, cv=10):
    """helper function for finding best model in parallel: evaluate model and return stat object. """

    random_search = RandomizedSearchCV(model_params['model'],
                                       param_distributions=model_params['params'],
                                       n_iter=20,
                                       n_jobs=-1,
                                       pre_dispatch='n_jobs',
                                       cv=cv,
                                       scoring='f1',
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
    """helper function to evaluate best model on held-out set. returns pipeline with best parameters"""

    best_model = results['best']['model']
    name = results['best']['name']

    selector = results['best']['selector']
    vectorizer = results['best']['vectorizer']

    if selector:
        transformedTestData = selector.transform(vectorizer.transform(X_eval))
    else:
        transformedTestData = vectorizer.transform(X_eval)

    y_pred = best_model.predict(transformedTestData)

    p, r, f1, _ = precision_recall_fscore_support(y_eval, y_pred, average='macro')
    acc = accuracy_score(y_eval, y_pred)
    clsr = classification_report(y_eval, y_pred)

    print("Testing best model on held-out test set:\n", name,
          results['best']['n-grams'], results['best']['fs'], "\n",
          'p={}\tr={}\tf1={}\tacc={}'.format(p, r, f1, acc))

    print("Classification report:\n{}".format(clsr))

    return Pipeline([('vectorizer', vectorizer), ('selector', selector), ('clf', best_model)])


def prepare_train_test(trainData, testData, trainLabels, rules=True, wordgram_range=None, feature_select=None,
                       chargram_range=None, genia_opts=None, min_df_char=0.001, min_df_word=0.001, max_df=1.0):
    """prepare training and test vectors, vectorizer and selector for validating classifiers"""

    print("Fitting vectorizer, preparing training and test data")

    vectorizer = transformers.vectorizer_dx(chargrams=chargram_range, min_df_char=min_df_char, wordgrams=wordgram_range,
                                            min_df_word=min_df_word, genia_opts=genia_opts, rules=rules, max_df=max_df)

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
