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
from helpers import num_rows

civic, abstracts = loaders.sentences_civic_abstracts()

print("CIViC sentences:", len(civic))
print("Abstract sentences:", len(abstracts))

piboso_other = loaders.sentences_piboso_other()
piboso_outcome = loaders.sentences_piboso_outcome()

print("PIBOSO sentences:", len(piboso_other))

P = civic
U = abstracts

P = random.sample(civic, 1000) + random.sample(piboso_outcome, 0)
U = random.sample(abstracts, 1000) + random.sample(P, 0)

# y_P = [1] * num_rows(P)
# y_U = [0] * num_rows(U)



# ------------------
# CR-SVM Test

print("\n\n"
      "-----------\n"
      "CR-SVM TEST\n"
      "-----------\n")

start_time = time.time()
crsvm = pu_two_step.cr_SVM(civic, abstracts, max_neg_ratio=0.1, noise_lvl=0.1, text=True)
print("\nTraining CR-SVM took %s seconds\n" % (time.time() - start_time))

civ_lab_sim = sorted(zip(crsvm.predict_proba(civic), civic), key=lambda x: x[0][0], reverse=True)
print("\ncos-rocchio civic prediction", sum([1 for x in civ_lab_sim if x[0][0] > 0.5]), "/", num_rows(civ_lab_sim))
print("civic top-12")
[print(x) for x in (civ_lab_sim[0:12])]
print("civic bot-12")
[print(x) for x in (civ_lab_sim[-12:])]

abs_lab_sim = sorted(zip(crsvm.predict_proba(abstracts), abstracts), key=lambda x: x[0][0], reverse=True)
print("\ncos-rocchio abstracts prediction", sum([1 for x in abs_lab_sim if x[0][0] > 0.5]), "/", num_rows(abs_lab_sim))
print("abstracts top-12")
[print(x) for x in (abs_lab_sim[0:12])]
print("abstracts bot-12")
[print(x) for x in (abs_lab_sim[-12:])]

oth_lab_sim = sorted(zip(crsvm.predict_proba(piboso_other), piboso_other), key=lambda x: x[0][0], reverse=True)
print("\ncos-rocchio piboso-other prediction", sum([1 for x in oth_lab_sim if x[0][0] > 0.5]), "/", num_rows(oth_lab_sim))
print("piboso other top-12")
[print(x) for x in (oth_lab_sim[0:12])]
print("piboso other bot-12")
[print(x) for x in (oth_lab_sim[-12:])]

out_lab_sim = sorted(zip(crsvm.predict_proba(piboso_outcome), piboso_outcome), key=lambda x: x[0][0], reverse=True)
print("\ncos-rocchio piboso-outcome prediction", sum([1 for x in out_lab_sim if x[0][0] > 0.5]), "/",
      num_rows(out_lab_sim))
print("piboso outcome top-12")
[print(x) for x in (out_lab_sim[0:12])]
print("piboso outcome bot-12")
[print(x) for x in (out_lab_sim[-12:])]
print("\n")

# ------------------
# standalone Rocchio Test

print("\n\n"
      "------------------------------\n"
      "STANDALONE COSINE-ROCCHIO TEST\n"
      "------------------------------\n")

start_time = time.time()
roc = pu_two_step.standalone_cos_rocchio(P, U, noise_lvl=0.5, text=True)
print("\nTraining Standalone Cosine-Rocchio took %s seconds\n" % (time.time() - start_time))

civ_lab_sim = sorted(zip(roc.predict_proba(civic), civic), key=itemgetter(0), reverse=True)
print("\ncos-rocchio civic prediction", sum([1 for x in civ_lab_sim if x[0] > 0.5]), "/", num_rows(civ_lab_sim))
print("civic top-12")
[print(x) for x in (civ_lab_sim[0:12])]
print("civic bot-12")
[print(x) for x in (civ_lab_sim[-12:])]

abs_lab_sim = sorted(zip(roc.predict_proba(abstracts), abstracts), key=itemgetter(0), reverse=True)
print("\ncos-rocchio abstracts prediction", sum([1 for x in abs_lab_sim if x[0] > 0.5]), "/", num_rows(abs_lab_sim))
print("abstracts top-12")
[print(x) for x in (abs_lab_sim[0:12])]
print("abstracts bot-12")
[print(x) for x in (abs_lab_sim[-12:])]

oth_lab_sim = sorted(zip(roc.predict_proba(piboso_other), piboso_other), key=itemgetter(0), reverse=True)
print("\ncos-rocchio piboso-other prediction", sum([1 for x in oth_lab_sim if x[0] > 0.5]), "/", num_rows(oth_lab_sim))
print("piboso other top-12")
[print(x) for x in (oth_lab_sim[0:12])]
print("piboso other bot-12")
[print(x) for x in (oth_lab_sim[-12:])]

out_lab_sim = sorted(zip(roc.predict_proba(piboso_outcome), piboso_outcome), key=itemgetter(0), reverse=True)
print("\ncos-rocchio piboso-outcome prediction", sum([1 for x in out_lab_sim if x[0] > 0.5]), "/",
      num_rows(out_lab_sim))
print("piboso outcome top-12")
[print(x) for x in (out_lab_sim[0:12])]
print("piboso outcome bot-12")
[print(x) for x in (out_lab_sim[-12:])]
print("\n")

# ------------------
# S-EM-Test

print("\n\n"
      "S-EM TEST\n"
      "---------\n")

start_time = time.time()

model = pu_two_step.s_EM(P, U, spy_ratio=0.1, tolerance=0.1, text=True)

print("\nS-EM took %s seconds\n" % (time.time() - start_time))

# ------------------
# I-EM-Test

print("\n\n"
      "I-EM TEST\n"
      "---------\n")

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
