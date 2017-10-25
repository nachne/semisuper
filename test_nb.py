from proba_label_nb import proba_label_MNB
from random import randint
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report as clsr
import pu_two_step
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import load_breast_cancer

print("\nNB test")
print("-------\n")

p_mnb = proba_label_MNB()

X = [[randint(0, 1), randint(0, 1), randint(0, 1), randint(0, 1), randint(0, 1), randint(0, 1)]
     for x in range(1000)]
y = [1. if (x[0] or x[1]) and not (x[4] and x[5]) else 0. for x in X]

# X = csr_matrix(X)

print("Building for evaluation")
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
model = p_mnb.fit(X_train, y_train)

print("Classification Report:\n")

y_pred = model.predict(X_test)
print(clsr(y_test, y_pred))

model = p_mnb.fit(X, y)

print("probs", model.predict_proba(X[:100]))
print("preds", model.predict(X[:100]))
print("labls", y[:100])

print(X[0])

print("\nNB test")
print("-------\n")

p_mnb = proba_label_MNB()

X = [[randint(0, 1), randint(0, 1), randint(0, 1), randint(0, 1), randint(0, 1), randint(0, 1)]
     for x in range(1000)]
y = [1. if (x[0] and x[1]) or (x[2] and x[1]) else 0. for x in X]

# X = csr_matrix(X)

print("Building for evaluation")
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
model = p_mnb.fit(X_train, y_train)

print("Classification Report:\n")

y_pred = model.predict(X_test)
print(clsr(y_test, y_pred))

model = p_mnb.fit(X, y)

print("preds", list(model.predict(X[:100])))
print("labls", y[:100])

print(X[0])

# TODO: schneller PU-Test mit richtigem Text (z.B. Vokabular einschränken)
print("\n---------------------------------------------------------------------------")
print("\nTwo-step technique tests (with numerical features and partly random labels)")
print("---------------------------------------------------------------------------\n")

featurenum = 40
quarter = int(featurenum / 4)

P = [([randint(0, 4) for _ in range(quarter)]
      + [randint(1, 2) for _ in range(2 * quarter)]
      + [1] * quarter
      )
     for _ in range(2000)]

P += [[randint(0,1) for _ in range(featurenum)] for _ in range(100)]

P = np.array([csr_matrix(p) for p in P])

U = [([randint(0, 2)
       for i in range(2 * quarter)]
      + ([0] * (2 * quarter))
      )
     for j in range(2000)]
U += [[randint(0, 2)
       for _ in range(featurenum)]
      for _ in range(500)]
U += [([randint(0, 2)
        for _ in range(2 * quarter)]
       + [randint(0, 2)
          for _ in range(2 * quarter)]
       )
      for _ in range(100)]
U += [([randint(0, 4) for _ in range(quarter)]
       + [randint(1, 2) for _ in range(2 * quarter)]
       + [1] * quarter
       )
      for _ in range(500)]
U += [[0]*featurenum] * 400
U += [[randint(0,1) for _ in range(featurenum)] for _ in range(1000)]

U = np.array([csr_matrix(u) for u in U])

print("done preparing")

# print(P[0])
# print(U[0])

print("\n---------------------------------------------------------------------------")
print("\nI_EM\n")
pu_two_step.i_EM(P, U, max_pos_ratio=1, max_imbalance=10, tolerance=0.1, text=False)

print("\n---------------------------------------------------------------------------")
print("\nS_EM\n")
pu_two_step.s_EM(P, U, max_pos_ratio=1, tolerance=0.1, text=False, spy_ratio=0.05)

print("\n---------------------------------------------------------------------------")
print("\nTesting with the Breast Cancer Dataset\n")

X, y = load_breast_cancer(return_X_y=True)

for row in range(np.shape(X)[0]):
    for col in range(np.shape(X)[1]):
        X[row,col] = 1 if X[row,col] > np.average(X,1)[col] else 0

negmask = np.ones(len(X), dtype=bool)
negmask[np.nonzero(y)] = False

TP = X[np.nonzero(y)]
TN = X[negmask]

P = np.concatenate((TP[:int(len(TP)/4)],
                   TN[-10:]))
U = np.concatenate((TP[int(len(TP)/4):],
                   TN[:-10]))

iem = pu_two_step.i_EM(P, U, max_pos_ratio=1, max_imbalance=10, tolerance=0.05, text=False)
sem = pu_two_step.s_EM(P, U, max_pos_ratio=1, tolerance=0.05, text=False, spy_ratio=0.15)

y_pred_iem = iem.predict(X)
y_pred_sem = sem.predict(X)

print("report I-EM with respect to ground truth:")
print(clsr(y, y_pred_iem))

print("report S-EM with respect to ground truth:")
print(clsr(y, y_pred_sem))
