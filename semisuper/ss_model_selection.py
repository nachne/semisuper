import multiprocessing as multi
import os
import time
from copy import deepcopy
from functools import partial
from itertools import product

import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from semisuper import transformers, ss_techniques
from semisuper.helpers import num_rows, concatenate

# TODO multiprocessing works on Linux when there aren't too many features, but not on macOS
PARALLEL = True  # os.sys.platform == "linux"


# ----------------------------------------------------------------
# Estimators and parameters to evaluate
# ----------------------------------------------------------------


def estimator_list():
    neg_svms = [{'name' : 'neglinSVC_C{}'.format(C),
                 'model': partial(ss_techniques.neg_self_training_clf, LinearSVC(C=C, class_weight='balanced'))}
                for C in np.arange(0.5, 1.01, 0.1)]

    neg_logits = [{'name' : 'neglinSVC_C{}'.format(C),
                   'model': partial(ss_techniques.neg_self_training_clf, LogisticRegression(solver='sag', C=C))}
                  for C in np.arange(0.5, 1.01, 0.1)]

    neg_sgds = [
        {'name' : 'negSGDmh',
         'model': partial(ss_techniques.neg_self_training_clf,
                          SGDClassifier(loss='modified_huber', class_weight='balanced'))},
        {'name' : 'negSGDsh',
         'model': partial(ss_techniques.neg_self_training_clf,
                          SGDClassifier(loss='squared_hinge', class_weight='balanced'))},
        {'name' : 'negSGDpc',
         'model': partial(ss_techniques.neg_self_training_clf,
                          SGDClassifier(loss='perceptron', class_weight='balanced'))}]

    others = [
        {'name': 'negNB_0.1', 'model': partial(ss_techniques.neg_self_training_clf, MultinomialNB(alpha=0.1))},
        {'name': 'negNB_1.0', 'model': partial(ss_techniques.neg_self_training_clf, MultinomialNB(alpha=1.0))},
        {'name': 'self-logit', 'model': ss_techniques.self_training},
        {'name': 'EM', 'model': ss_techniques.EM},
        {'name': 'kNN', 'model': ss_techniques.iterate_knn},
        {'name': 'label_propagation', 'model': ss_techniques.propagate_labels},
    ]

    return neg_svms + neg_logits + neg_sgds + others


def preproc_param_dict():
    d = {
        'df_min'        : [0.001, 0.005, 0.01],  # [0.001]
        'df_max'        : [1.0],
        'rules'         : [True, False],
        'genia_opts'    : [None,
                           {"pos": False, "ner": False},
                           {"pos": True, "ner": False},
                           {"pos": False, "ner": True},
                           {"pos": True, "ner": True}],
        'wordgram_range': [None, (1, 2), (1, 3), (1, 4)],
        'chargram_range': [None, (2, 4), (2, 5), (2, 6)],
        'feature_select': [
            # best: word (1,2)/(1,4), char (2,5)/(2,6), f 25%, rule True/False, SVC 1.0 / 0.75
            # w/o char: acc <= 0.80, w/o words: acc <= 0.84, U > 31%

            # ...breaks (too much RAM)
            # partial(transformers.factorization, 'TruncatedSVD', 200),
            # partial(transformers.factorization, 'LatentDirichletAllocation', 200),
            # partial(transformers.factorization, 'NMF', 200),

            transformers.IdentitySelector,
            partial(transformers.percentile_selector, 'chi2', 30),
            partial(transformers.percentile_selector, 'chi2', 25),
            partial(transformers.percentile_selector, 'chi2', 20),
            partial(transformers.percentile_selector, 'f', 30),
            partial(transformers.percentile_selector, 'f', 25),
            partial(transformers.percentile_selector, 'f', 20),
            partial(transformers.percentile_selector, 'mutual_info', 30),  # mutual information: worse than rest
            partial(transformers.percentile_selector, 'mutual_info', 25),
            partial(transformers.percentile_selector, 'mutual_info', 20),
            partial(transformers.select_from_l1_svc, 1.0, 1e-3),
            partial(transformers.select_from_l1_svc, 0.5, 1e-3),
            partial(transformers.select_from_l1_svc, 0.1, 1e-3),
        ]
    }

    return d


# ----------------------------------------------------------------
# Cross validation
# ----------------------------------------------------------------

def best_model_cross_val(P, N, U, fold=10):
    """determine best model, cross validate and return pipeline trained on all data"""

    print("\nFinding best model\n")

    best = get_best_model(P, N, U)['best']

    print("\nCross-validation\n")

    kf = KFold(n_splits=fold, shuffle=True)
    splits = zip(list(kf.split(P)), list(kf.split(N)))

    # TODO can't do this in parallel
    # if PARALLEL:
    #     with multi.Pool(min(fold, multi.cpu_count())) as p:
    #         stats = list(p.map(partial(eval_fold, best, P, N, U), enumerate(splits), chunksize=1))
    # else:
    #     stats = list(map(partial(eval_fold, best, P, N, U), enumerate(splits)))
    stats = list(map(partial(eval_fold, best, P, N, U), enumerate(splits)))

    mean_stats = np.mean(stats, 0)
    print("Cross-validation average: p {}, r {}, f1 {}, acc {}".format(
            mean_stats[0], mean_stats[1], mean_stats[2], mean_stats[3]))

    print("Retraining model on full data")

    vec, sel = best['vectorizer'], best['selector']
    vec.fit(concatenate((P, N, U)))
    P_, N_, U_ = [vec.transform(x) for x in [P, N, U]]

    y_pp = concatenate((np.ones(num_rows(P)), -np.ones(num_rows(N)), np.zeros(num_rows(U))))
    sel.fit(concatenate((P_, N_, U_)), y_pp)
    P_, N_, U_ = [(sel.transform(x)) for x in [P_, N_, U_]]

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

    # TODO fix this so it can run in parallel, remove print-debugging
    print("a")

    y_train_pp = concatenate((np.ones(num_rows(P_train)), -np.ones(num_rows(N_train)), np.zeros(num_rows(U))))

    print("b")

    pp = clone(Pipeline([('vectorizer', model_record['vectorizer']), ('selector', model_record['selector'])]))

    print("c")

    pp.fit(concatenate((P_train, N_train, U)), y_train_pp)

    print("d")

    P_, N_, U_, P_test_, N_test_ = [(pp.transform(x)) for x in [P_train, N_train, U, P_test, N_test]]

    print("e")

    model = model_record['untrained_model'](P_, N_, U_)

    print("f")

    y_pred = model.predict(concatenate((P_test_, N_test_)))
    y_test = concatenate((np.ones(num_rows(P_test_)), np.zeros(num_rows(N_test_))))

    print("g")

    pr, r, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("Fold no.", i, "acc", acc, "classification report:\n", classification_report(y_test, y_pred))
    return [pr, r, f1, acc]


# ----------------------------------------------------------------
# Model selection
# ----------------------------------------------------------------

def get_best_model(P_train, N_train, U_train, X_test=None, y_test=None):
    """Evaluate parameter combinations, save results and return object with stats of all models"""

    print("\nEvaluating parameter ranges for preprocessor and classifiers")

    if X_test is None or y_test is None:
        P_train, X_test_pos = train_test_split(P_train, test_size=0.2)
        N_train, X_test_neg = train_test_split(N_train, test_size=0.2)
        X_test = concatenate((X_test_pos, X_test_neg))
        y_test = concatenate((np.ones(num_rows(X_test_pos)), np.zeros(num_rows(X_test_neg))))

    X_train = concatenate((P_train, N_train, U_train))
    y_train_pp = concatenate((np.ones(num_rows(P_train)), -np.ones(num_rows(N_train)), np.zeros(num_rows(U_train))))

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
                                                                                 trainLabels=y_train_pp, rules=r,
                                                                                 wordgram_range=wordgram,
                                                                                 feature_select=fs,
                                                                                 chargram_range=chargram,
                                                                                 genia_opts=genia_opts,
                                                                                 min_df_char=df_min,
                                                                                 min_df_word=df_min, max_df=df_max)
                    if selector:
                        P_train_, N_train_, U_train_ = [(selector.transform(vectorizer.transform(x)))
                                                        for x in [P_train, N_train, U_train]]
                    else:
                        P_train_, N_train_, U_train_ = [(vectorizer.transform(x))
                                                        for x in [P_train, N_train, U_train]]

                    # fit models
                    if PARALLEL:
                        with multi.Pool(min(multi.cpu_count(), len(estimators))) as p:
                            iter_stats = list(p.map(partial(model_eval_record,
                                                            P_train_, N_train_, U_train_, X_test_, y_test),
                                                    estimators, chunksize=1))
                    else:
                        iter_stats = list(map(partial(model_eval_record,
                                                      P_train_, N_train_, U_train_, X_test_, y_test),
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

                    print("Evaluated words:", wordgram, "chars:", chargram,
                          "in %s seconds\n" % (time.time() - start_time))

                    print_reports(iter_stats)

    print_results(results)

    return results
    # return test_best(results, X_eval, y_eval)


# TODO obsolete, remove
def test_best(results, X_eval, y_eval):
    """helper function to evaluate best model on held-out set. returns full results of model selection"""

    best_model = results['best']['model']
    selector = results['best']['selector']
    vectorizer = results['best']['vectorizer']

    if selector:
        transformedTestData = (selector.transform(vectorizer.transform(X_eval)))
    else:
        transformedTestData = (vectorizer.transform(X_eval))

    y_pred = best_model.predict(transformedTestData)

    p, r, f, s = precision_recall_fscore_support(y_eval, y_pred, average='macro')
    acc = accuracy_score(y_eval, y_pred)

    print("Testing best model on held-out test set:\n", results['best']['name'],
          results['best']['n-grams'], results['best']['fs'], "\n",
          'p={}\tr={}\tf1={}\tacc={}'.format(p, r, f, acc))

    results['best']['eval stats'] = [p, r, f, s, acc]

    return results


def prepare_train_test(trainData, testData, trainLabels, rules=True, wordgram_range=None, feature_select=None,
                       chargram_range=None, genia_opts=None, min_df_char=0.001, min_df_word=0.001, max_df=1.0):
    """prepare training and test vectors, vectorizer and selector for validating classifiers"""

    print("Fitting vectorizer, preparing training and test data")

    vectorizer = transformers.vectorizer(chargrams=chargram_range, min_df_char=min_df_char, wordgrams=wordgram_range,
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


# ----------------------------------------------------------------
# Printing
# ----------------------------------------------------------------

def print_results(results):
    """helper function to print stat objects, starting with best model"""

    print("\n----------------------------------------------------------------\n")
    print("\nAll stats:\n")
    for r in results['all']:
        print_reports(r)

    print("\nBest:\n")
    best = results['best']
    print(best['name'], best['n-grams'], best['fs'],
          "\nrules, genia_opts", best['rules, genia_opts'], "df_min, df_max", best['df_min, df_max'],
          "\namount of U labelled as relevant:", best['U_ratio'],
          "\nstats: p={}\tr={}\tf1={}\tacc={}\t".format(best['p'], best['r'], best['f1'], best['acc']))

    print("\n----------------------------------------------------------------\n")
    return


def print_reports(i):
    """helper to print model stat object"""
    print(i[0]['n-grams'], i[0]['fs'],
          "\nrules, genia_opts", i[0]['rules, genia_opts'], "df_min, df_max", i[0]['df_min, df_max'])

    for m in i:
        print("\n{}:\tacc: {}, relevant ratio in U: {}, classification report:\n{}".format(
                m['name'], m['acc'], m['U_ratio'], m['clsr']))
    return


# ----------------------------------------------------------------
# helpers

def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)
