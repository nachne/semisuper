import semisuper.tests.test_uci_load_corpora as test_corpus
from numpy import concatenate, shape

from semisuper import pu_two_step, pu_biased_svm, ss_techniques, transformers
from semisuper.helpers import num_rows
from sklearn.metrics import classification_report as clsr, accuracy_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

import sys


# -------------------
# prepare test corpus
# -------------------

def prepare_corpus_pu(data_tuple):
    P, U, P_test, N_test = data_tuple

    y_P = [1] * num_rows(P)
    y_U = [0] * num_rows(U)

    X = concatenate((P, U))
    y = concatenate((y_P, y_U))

    y_P_test = [1] * num_rows(P_test)
    y_N_test = [0] * num_rows(N_test)

    X_test = concatenate((P_test, N_test))
    y_test = concatenate((y_P_test, y_N_test))

    vectorizer = transformers.vectorizer(chargrams=(2, 6), wordgrams=(1, 3), genia_opts=None, rules=False)

    print("Fitting vectorizer")
    vectorizer.fit(P)

    P = (vectorizer.transform(P))
    U = (vectorizer.transform(U))
    P_test = (vectorizer.transform(P_test))
    N_test = (vectorizer.transform(N_test))
    X = (vectorizer.transform(X))
    X_test = (vectorizer.transform(X_test))

    print("P:", shape(P), ", U:", shape(U))

    return P, U, X, y, X_test, y_test, P_test, y_P_test, N_test, y_N_test


def prepare_corpus_ss(data_tuple):
    P, N, U, P_test, N_test = data_tuple

    y_P = [1] * num_rows(P)
    y_N = [0] * num_rows(U)

    X = concatenate((P, N))
    y = concatenate((y_P, y_N))

    y_P_test = [1] * num_rows(P_test)
    y_N_test = [0] * num_rows(N_test)

    X_test = concatenate((P_test, N_test))
    y_test = concatenate((y_P_test, y_N_test))

    vectorizer = transformers.vectorizer(chargrams=(2, 6), min_df_char=0.01, wordgrams=(1, 3), min_df_word=0.01,
                                         genia_opts=None, rules=False)

    print("Fitting vectorizer")
    vectorizer.fit(P)

    P = (vectorizer.transform(P))
    N = (vectorizer.transform(N))
    U = (vectorizer.transform(U))
    P_test = (vectorizer.transform(P_test))
    N_test = (vectorizer.transform(N_test))
    X = (vectorizer.transform(X))
    X_test = (vectorizer.transform(X_test))

    print("P:", shape(P), ", N:", shape(N), "U:", shape(U))

    return P, N, U, X, y, X_test, y_test, P_test, y_P_test, N_test, y_N_test


# -------------------
# train models
# -------------------

def train_test_all_clfs_pu(data_tuple):
    P, U, X, y, X_test, y_test, P_test, y_P_test, N_test, y_N_test = data_tuple

    sup_mnb = MultinomialNB().fit(X, y)
    sup_linsvc = LinearSVC(C=0.1).fit(X, y)

    roc_svm = pu_two_step.roc_SVM(P, U)
    cr_svm = pu_two_step.cr_SVM(P, U, noise_lvl=0.4)

    i_em = pu_two_step.i_EM(P, U)
    s_em = pu_two_step.s_EM(P, U)

    roc_em = pu_two_step.roc_EM(P, U)
    spy_svm = pu_two_step.spy_SVM(P, U)

    biased_svm = pu_biased_svm.biased_SVM_weight_selection(P, U)

    print("\n\n-----------------------------------------------------------------------------")
    print("EVALUATION ON VALIDATION SET")
    print("-----------------------------------------------------------------------------\n")

    print("Supervised MNB:")
    y_pred = sup_mnb.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("Supervised SVC:")
    y_pred = sup_linsvc.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("Roc-SVM:")
    y_pred = roc_svm.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("CR-SVM:")
    y_pred = cr_svm.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("I-EM:")
    y_pred = i_em.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("S-EM:")
    y_pred = s_em.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("Roc-EM:")
    y_pred = roc_em.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("Spy-SVM:")
    y_pred = spy_svm.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("Biased-SVM:")
    y_pred = biased_svm.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    return


def train_test_all_clfs_ss(data_tuple):
    P, N, U, X, y, X_test, y_test, P_test, y_P_test, N_test, y_N_test = data_tuple

    sup_mnb = MultinomialNB().fit(concatenate((P, N)), [1] * num_rows(P) + [0] * num_rows(N))
    sup_linsvc = LinearSVC(C=0.1).fit(concatenate((P, N)), [1] * num_rows(P) + [0] * num_rows(N))

    neg_st_logreg = ss_techniques.neg_self_training(P, N, U)
    neg_st_sgd = ss_techniques.neg_self_training(P, N, U, clf=SGDClassifier(loss='modified_huber'))
    it_lin_svc = ss_techniques.iterate_linearSVC(P, N, U, 1.0)
    em = ss_techniques.EM(P, N, U)
    knn = ss_techniques.iterate_knn(P, N, U)

    print("\n\n-----------------------------------------------------------------------------")
    print("EVALUATION ON VALIDATION SET")
    print("-----------------------------------------------------------------------------\n")

    print("Supervised MNB:")
    y_pred = sup_mnb.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("Supervised SVC:")
    y_pred = sup_linsvc.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("Expanding negative set with Logistic Regression:")
    y_pred = neg_st_logreg.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("Expanding negative set with SGD:")
    y_pred = neg_st_sgd.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("Expanding negative set with linear SVC:")
    y_pred = it_lin_svc.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("Expectation-Maximisation:")
    y_pred = em.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("Self-training with k-NN:")
    y_pred = knn.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    return


# -------------------
# execute
# -------------------


newsgroups_ratio = 0.1
neg_noise = 0.01
pos_in_u = 0.4
neg_in_u = 0.9

print("---------------------------")
print("---------------------------")
print("SS LEARNING 20 NEWSGROUPS MAIN CATEGORIES")
print("---------------------------")
print("---------------------------")

print("ONE-VS-REST PER CATEGORY,", (100.0 * newsgroups_ratio), "% OF DATA")
i = 0
for tup in test_corpus.list_P_N_U_p_n_20_newsgroups(neg_noise=neg_noise,
                                                    pos_in_u=pos_in_u,
                                                    neg_in_u=neg_in_u,
                                                    test_size=0.2,
                                                    ratio=newsgroups_ratio):
    i += 1
    train_test_all_clfs_ss(prepare_corpus_ss(tup))

print("---------------------------")
print("---------------------------")
print("PU LEARNING 20 NEWSGROUPS MAIN CATEGORIES")
print("---------------------------")
print("---------------------------")

print("ONE-VS-REST PER CATEGORY,", (100.0 * newsgroups_ratio), "% OF DATA")
i = 0
for tup in test_corpus.list_P_U_p_n_20_newsgroups(neg_noise=neg_noise,
                                                  pos_in_u=pos_in_u,
                                                  test_size=0.2,
                                                  ratio=newsgroups_ratio):
    print("\nP := NEWSGROUP CATEGORY", i, "\n")
    i += 1
    train_test_all_clfs_pu(prepare_corpus_pu(tup))

print("------------------------------------------------------------------------------------------------------------")
sys.exit(0)

print("---------------------------")
print("---------------------------")
print("PU AMAZON-IMDB-YELP CORPUS")
print("---------------------------")
print("---------------------------")

train_test_all_clfs(prepare_corpus(test_corpus.P_U_p_n_amazon(neg_noise=neg_noise,
                                                              pos_in_u=pos_in_u,
                                                              test_size=0.2)))

print("---------------------------")
print("---------------------------")
print("PU SMS SPAM CORPUS")
print("---------------------------")
print("---------------------------")

train_test_all_clfs(prepare_corpus(test_corpus.P_U_p_n_sms_spam(neg_noise=neg_noise,
                                                                pos_in_u=pos_in_u,
                                                                test_size=0.2)))

print("---------------------------")
print("---------------------------")
print("PU UCI SENTENCE CORPUS")
print("---------------------------")
print("---------------------------")

train_test_all_clfs(prepare_corpus(test_corpus.P_U_p_n_uci_sentences(neg_noise=neg_noise,
                                                                     pos_in_u=pos_in_u,
                                                                     test_size=0.2)))
