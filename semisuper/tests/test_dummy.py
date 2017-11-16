from semisuper import transformers, basic_pipeline, loaders
from sklearn.linear_model import SGDClassifier
import random
import pandas as pd
import sys

civic, abstracts = loaders.sentences_civic_abstracts()

print("CIViC sentences:", len(civic))
print("abstract sentences:", len(abstracts))

piboso_other = loaders.sentences_piboso_other()
piboso_outcome = loaders.sentences_piboso_outcome()

print("PIBOSO sentences:", len(piboso_other))

pos = civic
neg = abstracts

# pos = random.sample(civic, 2000) + random.sample(piboso_outcome, 0)
# neg = random.sample(abstracts, 2000) + random.sample(piboso_other, 0)

X = pos + neg
y = ["pos"] * len(pos) + ["neg"] * len(neg)

# ----------------------------------------------------------------
# CIViC vs abstracts normal classifier test

print("\n\n"
      "----------------------------------\n"
      "SUPERVISED CIVIC VS ABSTRACTS TEST\n"
      "----------------------------------\n")

# comment out for quick testing of existing model
model = basic_pipeline.build_classifier(X, y, classifier=SGDClassifier,
                                        words=True, chars=True, chargram_range=(1,3))

print("Most informative features per class:")
print(basic_pipeline.show_most_informative_features(model))

# ----------------------------------------------------------------
sys.exit(0)

# unpickle classifier
# with open('../pickles/dummy_clf.pickle', 'rb') as f:
#     model = pickle.load(f)

preppy = transformers.TokenizePreprocessor()

lab_civ = pd.DataFrame(data={"Label" : model.predict(civic),
                             "Text"  : civic,
                             "Tokens": preppy.transform(civic)},
                       columns=["Label", "Text", "Tokens"])

lab_civ.to_csv("../output/labelled_dummy_civic.csv")

lab_abs = pd.DataFrame(data={"Label" : model.predict(abstracts),
                             "Text"  : abstracts,
                             "Tokens": preppy.transform(abstracts)},
                       columns=["Label", "Text", "Tokens"])

lab_abs.to_csv("../output/labelled_dummy_abstracts.csv")

lab_oth = pd.DataFrame(data={"Label" : model.predict(piboso_other),
                             "Text"  : piboso_other,
                             "Tokens": preppy.transform(piboso_other)},
                       columns=["Label", "Text", "Tokens"])

lab_oth.to_csv("../output/labelled_dummy_other.csv")

lab_out = pd.DataFrame(data={"Label" : model.predict(piboso_outcome),
                             "Text"  : piboso_outcome,
                             "Tokens": preppy.transform(piboso_outcome)},
                       columns=["Label", "Text", "Tokens"])

lab_out.to_csv("../output/labelled_dummy_outcome.csv")

# save sentences with predicted labels to csv


# ----------------------------------------------------------------

print("civic: prediction", sum(lab_civ["Label"].values), "/", len(civic))

print("abstracts: prediction", sum(lab_abs["Label"].values), "/", len(lab_abs))

yhat3 = model.predict(loaders.sentences_piboso_pop_bg_oth())
print("piboso pop bg oth: prediction", sum(yhat3), "/", len(yhat3))

yhat4 = model.predict(loaders.sentences_piboso_other())
print("piboso other: prediction", sum(yhat4), "/", len(yhat4))

yhat5 = model.predict(loaders.sentences_piboso_outcome())
print("piboso outcome: prediction", sum(yhat5), "/", len(yhat5))
