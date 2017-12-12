import sys

from semisuper import loaders, transformers
from sklearn.linear_model import SGDClassifier
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
model = SGDClassifier().fit(X, y)


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

