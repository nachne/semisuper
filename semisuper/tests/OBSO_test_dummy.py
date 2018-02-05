from __future__ import absolute_import, division, print_function

import sys

from semisuper import loaders, transformers, helpers
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd

civic, abstracts = loaders.sentences_civic_abstracts()
hocpos, hocneg = loaders.sentences_HoC()
piboso_other = loaders.sentences_piboso_other()
piboso_outcome = loaders.sentences_piboso_outcome()

print("CIViC sentences:", len(civic))
print("Abstract sentences:", len(abstracts))
print("HoC positive sentences:", len(hocpos))
print("HoC negative sentences:", len(hocneg))
print("PIBOSO outcome sentences:", len(piboso_outcome))
print("PIBOSO other sentences:", len(piboso_other))


def test_supervised(P, N, clf=LinearSVC(class_weight='balanced'),
                    wordgrams=(1, 1), chargrams=None, stats=None, min_df=1):
    X_ = helpers.concatenate((P, N))
    y = helpers.concatenate(([1] * helpers.num_rows(P), [0] * helpers.num_rows(N)))
    X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.2)

    vec = transformers.vectorizer(chargrams=chargrams, wordgrams=wordgrams,
                                  min_df_word=min_df, min_df_char=min_df,
                                  rules=False, stats=stats)
    X_train = vec.fit_transform(X_train)
    X_test = vec.transform(X_test)

    model = clf
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("acc:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return


def test_supervised(P, N, clf=LinearSVC(class_weight='balanced'),
                    wordgrams=(1, 1), chargrams=None, stats=None, min_df=1):
    X_ = helpers.concatenate((P, N))
    y = helpers.concatenate(([1] * helpers.num_rows(P), [0] * helpers.num_rows(N)))

    vec = transformers.vectorizer(chargrams=chargrams, wordgrams=wordgrams,
                                  min_df_word=min_df, min_df_char=min_df,
                                  rules=False, stats=stats)
    X = vec.fit_transform(X_, y)

    scores = cross_val_score(clf, X, y, cv=10, scoring='f1_weighted')

    print(scores, "avg:", np.average(scores))


for min_df in [0.001, 0.002, 0.005, 0.01, 1]:
    for clf in [LinearSVC(class_weight='balanced')]:  # [, LogisticRegression, MultinomialNB,
        # KNeighborsClassifier, SGDClassifier]:
        print("\n--------------------------------\n{}\n--------------------------------".format(clf))
        print("min_df:", min_df)
        # print("\nCIViC + abstracts   VS   HoC_p + HoC_n")
        # test_supervised(helpers.concatenate((civic, abstracts)), helpers.concatenate((hocpos, hocneg)))

        print("\nCIViC + HoC_p   VS   HoC_n")
        test_supervised(helpers.concatenate((civic, hocpos)), hocneg,
                        stats="length", wordgrams=(1, 4), chargrams=(2, 6),
                        min_df=min_df)

        # print("\none vs one")
        #
        # print("\nCIViC   VS   abstracts")
        # test_supervised(civic, abstracts)
        #
        # print("\nCIViC   VS   HoC_n")
        # test_supervised(civic, hocneg)
        #
        # print("\nCIViC   VS   HoC_p")
        # test_supervised(civic, hocpos)
        #
        # print("\nabstracts   VS   HoC_n")
        # test_supervised(abstracts, hocneg)
        #
        # print("\nabstracts   VS   HoC_p")
        # test_supervised(abstracts, hocpos)

        print("\nHoC_p   VS   HoC_n")
        test_supervised(hocpos, hocneg,
                        stats="length", wordgrams=(1, 4), chargrams=(2, 6),
                        min_df=min_df)

sys.exit(0)




pos = civic + hocpos
neg = abstracts + hocneg

# pos = random.sample(civic, 2000) + random.sample(piboso_outcome, 0)
# neg = random.sample(abstracts, 2000) + random.sample(piboso_other, 0)

X = pos + neg
y = [1] * len(pos) + [0] * len(neg)

vec = transformers.vectorizer()

X = vec.fit_transform(X, y)

# ----------------------------------------------------------------
# CIViC vs abstracts normal classifier test

print("\n\n"
      "----------------------------------\n"
      "SUPERVISED CIVIC VS ABSTRACTS TEST\n"
      "----------------------------------\n")

# comment out for quick testing of existing model
model = LinearSVC().fit(X, y)

# ----------------------------------------------------------------

print("civic: prediction", np.sum(model.predict(vec.transform(civic))), "/", len(civic))

print("abstracts: prediction", np.sum(model.predict(vec.transform(abstracts))), "/", len(abstracts))

print("hocpos: prediction", np.sum(model.predict(vec.transform(hocpos))), "/", len(hocpos))

print("hocneg: prediction", np.sum(model.predict(vec.transform(hocneg))), "/", len(hocneg))

# ----------------------------------------------------------------

sys.exit(0)

# unpickle classifier
# with open('../pickles/dummy_clf.pickle', 'rb') as f:
#     model = pickle.load(f)

preppy = transformers.TokenizePreprocessor()

lab_civ = pd.DataFrame(data={"Label" : model.predict(vec.transform(civic)),
                             "Text"  : civic,
                             "Tokens": preppy.transform(civic)},
                       columns=["Label", "Text", "Tokens"])

lab_abs = pd.DataFrame(data={"Label" : model.predict(vec.transform(abstracts)),
                             "Text"  : abstracts,
                             "Tokens": preppy.transform(abstracts)},
                       columns=["Label", "Text", "Tokens"])

lab_hocpos = pd.DataFrame(data={"Label" : model.predict(vec.transform(hocpos)),
                                "Text"  : hocpos,
                                "Tokens": preppy.transform(hocpos)},
                          columns=["Label", "Text", "Tokens"])

lab_hocneg = pd.DataFrame(data={"Label" : model.predict(vec.transform(hocneg)),
                                "Text"  : hocneg,
                                "Tokens": preppy.transform(hocneg)},
                          columns=["Label", "Text", "Tokens"])

# lab_oth = pd.DataFrame(data={"Label" : model.predict(vec.transform(piboso_other)),
#                              "Text"  : piboso_other,
#                              "Tokens": preppy.transform(piboso_other)},
#                        columns=["Label", "Text", "Tokens"])
#
#
# lab_out = pd.DataFrame(data={"Label" : model.predict(vec.transform(piboso_outcome)),
#                              "Text"  : piboso_outcome,
#                              "Tokens": preppy.transform(piboso_outcome)},
#                        columns=["Label", "Text", "Tokens"])


# save sentences with predicted labels to csv
# lab_civ.to_csv("../output/labelled_dummy_civic.csv")
# lab_abs.to_csv("../output/labelled_dummy_abstracts.csv")
# lab_oth.to_csv("../output/labelled_dummy_other.csv")
# lab_out.to_csv("../output/labelled_dummy_outcome.csv")
