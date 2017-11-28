import semisuper.tests.load_test_corpora as test_corpus
from numpy import concatenate, shape
from semisuper import pu_two_step, pu_biased_svm, pu_one_class_svm, basic_pipeline
from semisuper.helpers import densify, num_rows
from sklearn.metrics import classification_report as clsr, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


# -------------------
# prepare test corpus
# -------------------

def prepare_corpus(data_tuple):
    P, U, P_test, N_test = data_tuple

    y_P = [1] * num_rows(P)
    y_U = [0] * num_rows(U)

    X = concatenate((P, U))
    y = concatenate((y_P, y_U))

    y_P_test = [1] * num_rows(P_test)
    y_N_test = [0] * num_rows(N_test)

    X_test = concatenate((P_test, N_test))
    y_test = concatenate((y_P_test, y_N_test))

    vectorizer = basic_pipeline.vectorizer(words=True, wordgram_range=(1, 4), chars=False, chargram_range=(2, 6),
                                           rules=False, lemmatize=False)

    print("Fitting vectorizer")
    vectorizer.fit(P)

    P = densify(vectorizer.transform(P))
    U = densify(vectorizer.transform(U))
    P_test = densify(vectorizer.transform(P_test))
    N_test = densify(vectorizer.transform(N_test))
    X = densify(vectorizer.transform(X))
    X_test = densify(vectorizer.transform(X_test))

    print("P:", shape(P), ", U:", shape(U))

    return P, U, X, y, X_test, y_test, P_test, y_P_test, N_test, y_N_test


# -------------------
# train models
# -------------------

def train_test_all_clfs(data_tuple):
    P, U, X, y, X_test, y_test, P_test, y_P_test, N_test, y_N_test = data_tuple

    sup_mnb = basic_pipeline.train_clf(X, y, MultinomialNB())
    sup_linsvc = basic_pipeline.train_clf(X, y, LinearSVC(C=0.1))

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

    print("---------------------------")
    print("Supervised MNB:")
    y_pred = sup_mnb.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("---------------------------")
    print("Supervised SVC:")
    y_pred = sup_linsvc.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("---------------------------")
    print("Roc-SVM:")
    y_pred = roc_svm.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("---------------------------")
    print("CR-SVM:")
    y_pred = cr_svm.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("---------------------------")
    print("I-EM:")
    y_pred = i_em.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("---------------------------")
    print("S-EM:")
    y_pred = s_em.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("---------------------------")
    print("Roc-EM:")
    y_pred = roc_em.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("---------------------------")
    print("Spy-SVM:")
    y_pred = spy_svm.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    print("---------------------------")
    print("Biased-SVM:")
    y_pred = biased_svm.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(clsr(y_test, y_pred))

    return


# -------------------
# execute
# -------------------

neg_noise = 0.02
pos_in_u = 0.4

print("---------------------------")
print("---------------------------")
print("20 NEWSGROUPS")
print("---------------------------")
print("---------------------------")

ratio = 1.0
print("ONE-VS-REST PER CATEGORY,", (100.0*ratio), "% OF DATA")
i = 0
for tup in test_corpus.list_P_U_p_n_20_newsgroups(neg_noise=neg_noise,
                                                  pos_in_u=pos_in_u,
                                                  test_size=0.2,
                                                  ratio=ratio):
    print("---------------------------")
    print("P := NEWSGROUP CATEGORY", i)
    i += 1
    train_test_all_clfs(prepare_corpus(tup))

print("---------------------------")
print("---------------------------")
print("AMAZON-IMDB-YELP CORPUS")
print("---------------------------")
print("---------------------------")

train_test_all_clfs(prepare_corpus(test_corpus.P_U_p_n_amazon(neg_noise=neg_noise,
                                                              pos_in_u=pos_in_u,
                                                              test_size=0.2)))

print("---------------------------")
print("---------------------------")
print("SMS SPAM CORPUS")
print("---------------------------")
print("---------------------------")

train_test_all_clfs(prepare_corpus(test_corpus.P_U_p_n_sms_spam(neg_noise=neg_noise,
                                                                pos_in_u=pos_in_u,
                                                                test_size=0.2)))

print("---------------------------")
print("---------------------------")
print("UCI SENTENCE CORPUS")
print("---------------------------")
print("---------------------------")

train_test_all_clfs(prepare_corpus(test_corpus.P_U_p_n_uci_sentences(neg_noise=neg_noise,
                                                                     pos_in_u=pos_in_u,
                                                                     test_size=0.2)))
