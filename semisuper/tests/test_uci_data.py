import semisuper.tests.load_test_corpus as test_corpus
from semisuper import pu_two_step, pu_biased_svm, pu_ranking, pu_one_class_svm, dummy_pipeline
from numpy import concatenate, arange
from sklearn.metrics import classification_report as clsr

# -------------------
# prepare test corpus
# -------------------

P, U, P_test, N_test = test_corpus.P_U_p_n_uci()

y_P = [1] * len(P)
y_U = [0] * len(U)

X = concatenate((P, U))
y = concatenate((y_P, y_U))

y_P_test = [1] * len(P_test)
y_N_test = [0] * len(N_test)

X_test = concatenate((P_test, N_test))
y_test = concatenate((y_P_test, y_N_test))

print("P:", len(P), ", U:", len(U))

# -------------------
# train models
# -------------------

print("\n\n---------------------------")
print("Training dummy classifier")
print("---------------------------\n")
dummy = dummy_pipeline.build_and_evaluate(X, y)

print("\n\n---------------------------")
print("Training one-class SVM")
print("---------------------------\n")
one_class = pu_one_class_svm.one_class_svm(P, X_test)

print("\n\n---------------------------")
print("Training cos-roc classifier")
print("---------------------------\n")
cos_roc = pu_two_step.standalone_cos_rocchio(P, U)

print("\n\n---------------------------")
print("Training roc-SVM classifier")
print("---------------------------\n")
roc_svm = pu_two_step.roc_SVM(P, U)

print("\n\n---------------------------")
print("Training CR_SVM classifier")
print("---------------------------\n")
cr_svm = pu_two_step.cr_SVM(P, U)

print("\n\n---------------------------")
print("Training I-EM classifier")
print("---------------------------\n")
i_em = pu_two_step.i_EM(P, U)

print("\n\n---------------------------")
print("Training S-EM classifier")
print("---------------------------\n")
s_em = pu_two_step.s_EM(P, U)


print("\n\n---------------------------")
print("Training Biased-SVM")
print("---------------------------\n")
biased_svm = pu_biased_svm.biased_SVM_weight_selection(P, U,
                                                       # Cs=[10 ** x for x in range(-4, 5, 4)],
                                                       Cs_neg=arange(0.01, 0.63, 0.32),
                                                       Cs_pos_factors=range(1, 2200, 200),
                                                       text=True)


print("\n\n-----------------------------------------------------------------------------")
print("EVALUATION ON VALIDATION SET")
print("-----------------------------------------------------------------------------\n")

print("---------------------------")
print("Dummy:")
print(clsr(y_test, dummy.predict(X_test)))

print("---------------------------")
print("One-Class SVM:")
print(clsr([1]*len(P_test)+[-1]*len(N_test), one_class.predict(X_test)))

print("---------------------------")
print("Cos-Roc:")
print(clsr(y_test, cos_roc.predict(X_test)))

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
print("Biased-SVM:")
print(clsr(y_test, biased_svm.predict(X_test)))