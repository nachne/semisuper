from simple_nb import proba_label_MNB
from random import randint
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report as clsr
import pu_two_step
import numpy as np
from scipy.sparse import csr_matrix


print("\nNB test")
print("-------\n")

p_mnb = proba_label_MNB()

X = [[randint(0, 1), randint(0, 1), randint(0, 1), randint(0, 1), randint(0, 1), randint(0, 1)]
     for x in range(1000)]
y = [1.0 if (x[0] or x[1]) and not (x[4] and x[5]) else 0.0 for x in X]

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
y = [1.0 if (x[0] and x[1]) or (x[2] and x[1]) else 0.0 for x in X]

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

# TODO: schneller PU-Test mit richtigem Text (z.B. Vokabular einschränken)
print("\nPU NB I-EM test")
print("---------------\n")

featurenum = 8
quarter = int(featurenum/4)

P = [([randint(0, 4) for i in range(quarter)]
      + [randint(1, 2) for i in range(2*quarter)]
      + [1] * quarter
      )
     for j in range(500)]

P = np.array([csr_matrix(p) for p in P])

U = [([randint(0, 2)
      for i in range(2*quarter)]
      + ([0] * (2*quarter))
      )
     for j in range(1500)]
U += [[0] * featurenum] * 500
U += [([randint(0, 2)
      for i in range(2*quarter)]
      + [randint(0, 2)
      for i in range(2*quarter)]
      )
     for j in range(100)]
U += [([randint(0, 4) for i in range(quarter)]
      + [randint(1, 2) for i in range(2*quarter)]
      + [1] * quarter
      )
     for j in range(100)]

U = np.array([csr_matrix(u) for u in U])

print("done preparing")

# print(P[0])
# print(U[0])

pu_two_step.expectation_maximization(P, U, max_pos_ratio=1, max_imbalance=10, tolerance=0.1, text=False)
