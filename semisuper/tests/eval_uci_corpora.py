import semisuper.tests.load_test_corpora as test_corpus
from numpy import concatenate
from semisuper import pu_two_step, pu_biased_svm, pu_one_class_svm, basic_pipeline
from sklearn.metrics import classification_report as clsr


# -------------------
# prepare test corpus
# -------------------

def prepare_corpus(data_tuple):
    P, U, P_test, N_test = data_tuple

    y_P = [1] * len(P)
    y_U = [0] * len(U)

    X = concatenate((P, U))
    y = concatenate((y_P, y_U))

    y_P_test = [1] * len(P_test)
    y_N_test = [0] * len(N_test)

    X_test = concatenate((P_test, N_test))
    y_test = concatenate((y_P_test, y_N_test))

    print("P:", len(P), ", U:", len(U))

    return P, U, X, y, X_test, y_test, P_test, y_P_test, N_test, y_N_test


# -------------------
# train models
# -------------------

def train_test_all_clfs(data_tuple):
    P, U, X, y, X_test, y_test, P_test, y_P_test, N_test, y_N_test = data_tuple

    print("\n\n---------------------------")
    print("Training dummy classifier")
    print("---------------------------\n")
    dummy = basic_pipeline.train_clf(X, y, None)

    print("\n\n---------------------------")
    print("Training one-class SVM")
    print("---------------------------\n")
    one_class = pu_one_class_svm.one_class_svm(P, X_test)

    print("\n\n---------------------------")
    print("Training roc-SVM classifier")
    print("---------------------------\n")
    roc_svm = pu_two_step.roc_SVM(P, U)

    print("\n\n---------------------------")
    print("Training CR_SVM classifier")
    print("---------------------------\n")
    cr_svm = pu_two_step.cr_SVM(P, U, noise_lvl=0.4)

    print("\n\n---------------------------")
    print("Training I-EM classifier")
    print("---------------------------\n")
    i_em = pu_two_step.i_EM(P, U)

    print("\n\n---------------------------")
    print("Training S-EM classifier")
    print("---------------------------\n")
    s_em = pu_two_step.s_EM(P, U)

    print("\n\n---------------------------")
    print("Training Roc-EM classifier")
    print("---------------------------\n")
    roc_em = pu_two_step.roc_EM(P, U)

    print("\n\n---------------------------")
    print("Training Spy-SVM classifier")
    print("---------------------------\n")
    spy_svm = pu_two_step.spy_SVM(P, U)

    print("\n\n---------------------------")
    print("Training Biased-SVM")
    print("---------------------------\n")
    biased_svm = pu_biased_svm.biased_SVM_weight_selection(P, U)

    print("\n\n-----------------------------------------------------------------------------")
    print("EVALUATION ON VALIDATION SET")
    print("-----------------------------------------------------------------------------\n")

    print("---------------------------")
    print("Dummy:")
    print(clsr(y_test, dummy.predict(X_test)))

    print("---------------------------")
    print("One-Class SVM:")
    print(clsr([1] * len(P_test) + [-1] * len(N_test), one_class.predict(X_test)))

    print("---------------------------")
    print("Roc-SVM:")
    print(clsr(y_test, roc_svm.predict(X_test)))

    print("---------------------------")
    print("CR-SVM:")
    print(clsr(y_test, cr_svm.predict(X_test)))

    print("---------------------------")
    print("I-EM:")
    print(clsr(y_test, i_em.predict(X_test)))

    print("---------------------------")
    print("S-EM:")
    print(clsr(y_test, s_em.predict(X_test)))

    print("---------------------------")
    print("Roc-EM:")
    print(clsr(y_test, roc_em.predict(X_test)))

    print("---------------------------")
    print("Spy-SVM:")
    print(clsr(y_test, spy_svm.predict(X_test)))

    print("---------------------------")
    print("Biased-SVM:")
    print(clsr(y_test, biased_svm.predict(X_test)))

    return


# -------------------
# execute
# -------------------

neg_noise = 0.02
pos_in_u = 0.5

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
print("UCI CORPUS")
print("---------------------------")
print("---------------------------")

train_test_all_clfs(prepare_corpus(test_corpus.P_U_p_n_uci(neg_noise=neg_noise,
                                                           pos_in_u=pos_in_u,
                                                           test_size=0.2)))
