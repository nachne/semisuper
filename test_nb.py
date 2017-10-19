from simple_nb import proba_label_MNB
from random import randint
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report as clsr

p_mnb = proba_label_MNB()

X = [[randint(0, 1), randint(0, 1), randint(0, 1), randint(0, 1), randint(0, 1), randint(0, 1)] for x in range(100000)]
y = [1.0 if x[1] else 0.0 for x in X]


print("Building for evaluation")
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
model = p_mnb.fit(X_train, y_train)

print("Classification Report:\n")

y_pred = model.predict(X_test)
print(clsr(y_test, y_pred))

model = p_mnb.fit(X, y)

print("labls", y[:100])
print("probs", model.predict_proba(X[:100]))
print("preds", model.predict(X[:100]))
