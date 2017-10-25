import transformers
import dummy_pipeline
from pu_one_class_svm import one_class_svm
from pu_ranking import ranking_cos_sim
import pickle
from sklearn.feature_extraction import DictVectorizer
import loaders
import random
from operator import itemgetter
import re
import pandas as pd
import pu_two_step
import time
import sys

civic, abstracts = loaders.sentences_civic_abstracts()

print("CIViC sentences:", len(civic))
print("abstract sentences:", len(abstracts))

piboso_other = loaders.sentences_piboso_other()
piboso_outcome = loaders.sentences_piboso_outcome()

print("PIBOSO sentences:", len(piboso_other))

P = civic
U = abstracts

P = random.sample(civic, 1000) + random.sample(piboso_outcome, 0)
U = random.sample(abstracts, 1000) + random.sample(P, 0)


# ------------------
# fake S-EM-Test

RN = random.sample(abstracts, 500)

model = pu_two_step.run_EM_with_reliable_negs(P, U, RN, tolerance=0.1)

# ------------------
# I-EM-Test

print("\n\n"
      "I-EM TEST\n"
      "---------\n")

U += RN

start_time = time.time()

model = pu_two_step.i_EM(P, U, max_pos_ratio=0.5, max_imbalance=1.0, tolerance=0.15, text=True)

print("\nEM took %s seconds\n" % (time.time() - start_time))

# print(dummy_pipeline.show_most_informative_features(model))

# ----------------------------------------------------------------
sys.exit(0)
# ----------------------------------------------------------------


preppy = transformers.BasicPreprocessor()

lab_civ = pd.DataFrame(data={"Label" : model.predict(civic),
                             "Text"  : civic,
                             "Tokens": preppy.transform(civic)},
                       columns=["Label", "Text", "Tokens"])

lab_civ.to_csv("./labelled_i-em_civic.csv")

lab_abs = pd.DataFrame(data={"Label" : model.predict(abstracts),
                             "Text"  : abstracts,
                             "Tokens": preppy.transform(abstracts)},
                       columns=["Label", "Text", "Tokens"])

lab_abs.to_csv("./labelled_i-em_abstracts.csv")

lab_oth = pd.DataFrame(data={"Label" : model.predict(piboso_other),
                             "Text"  : piboso_other,
                             "Tokens": preppy.transform(piboso_other)},
                       columns=["Label", "Text", "Tokens"])

lab_oth.to_csv("./labelled_i-em_other.csv")

lab_out = pd.DataFrame(data={"Label" : model.predict(piboso_outcome),
                             "Text"  : piboso_outcome,
                             "Tokens": preppy.transform(piboso_outcome)},
                       columns=["Label", "Text", "Tokens"])

lab_out.to_csv("./labelled_i-em_outcome.csv")

# save sentences with predicted labels to csv


print("civic: prediction", sum(lab_civ["Label"].values), "/", len(civic))

print("abstracts: prediction", sum(lab_abs["Label"].values), "/", len(abstracts))


print("piboso other: prediction", sum(lab_oth["Label"].values), "/", len(piboso_other))

print("piboso outcome: prediction", sum(lab_out["Label"].values), "/", len(piboso_outcome))

yhat3 = model.predict(loaders.sentences_piboso_pop_bg_oth())
print("piboso pop bg oth: prediction", sum(yhat3), "/", len(yhat3))
