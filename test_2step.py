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

P = random.sample(civic, 4000) + random.sample(piboso_outcome, 0)
U = random.sample(abstracts, 4000) + random.sample(P, 0)

# ------------------
# standalone Rocchio Test

print("\n\n"
      "------------------------------\n"
      "STANDALONE COSINE-ROCCHIO TEST\n"
      "------------------------------\n")

start_time = time.time()
roc = pu_two_step.standalone_cos_rocchio(P, U, noise_lvl=0.5, text=True)
print("\nTraining Standalone Cosine-Rocchio took %s seconds\n" % (time.time() - start_time))


def sort_cos_roc(sentences):
    return sorted(zip(roc.predict_proba(sentences),
                      sentences),
                  key=lambda x: x[0],
                  reverse=True)


def top_bot_12_cos_roc(predictions, name):
    print("\ncos-rocchio", name, "prediction", sum([1 for x in predictions if x[0] > 0.5]), "/",
          num_rows(predictions))
    print(name, "top-12")
    [print(x) for x in (predictions[0:12])]
    print(name, "bot-12")
    [print(x) for x in (predictions[-12:])]


civ_lab_sim = sort_cos_roc(civic)
top_bot_12_cos_roc(civ_lab_sim, "civic")

abs_lab_sim = sort_cos_roc(abstracts)
top_bot_12_cos_roc(abs_lab_sim, "abstracts")

oth_lab_sim = sort_cos_roc(piboso_other)
top_bot_12_cos_roc(oth_lab_sim, "piboso-other")

out_lab_sim = sort_cos_roc(piboso_outcome)
top_bot_12_cos_roc(out_lab_sim, "piboso-outcome")

# ------------------
# ROC-SVM Test

print("\n\n"
      "------------\n"
      "ROC-SVM TEST\n"
      "------------\n")

start_time = time.time()
rocsvm = pu_two_step.roc_SVM(P, U, max_neg_ratio=0.1, text=True)
print("\nTraining ROC-SVM took %s seconds\n" % (time.time() - start_time))

def sort_roc_svm(sentences):
    return sorted(zip(rocsvm.predict_proba(sentences),
                      sentences),
                  key=lambda x: x[0][1],
                  reverse=True)

def top_bot_12_cr_svm(predictions, name):
    print("\nroc-svm", name, "prediction", sum([1 for x in predictions if x[0][1] > 0.5]), "/",
          num_rows(predictions))
    print(name, "top-12")
    [print(x) for x in (predictions[0:12])]
    print(name, "bot-12")
    [print(x) for x in (predictions[-12:])]


civ_lab_sim = sort_roc_svm(civic)
top_bot_12_cr_svm(civ_lab_sim, "civic")

abs_lab_sim = sort_roc_svm(abstracts)
top_bot_12_cr_svm(abs_lab_sim, "abstracts")

oth_lab_sim = sort_roc_svm(piboso_other)
top_bot_12_cr_svm(oth_lab_sim, "piboso-other")

out_lab_sim = sort_roc_svm(piboso_outcome)
top_bot_12_cr_svm(out_lab_sim, "piboso-outcome")


# ------------------
# CR-SVM Test

print("\n\n"
      "-----------\n"
      "CR-SVM TEST\n"
      "-----------\n")

start_time = time.time()
crsvm = pu_two_step.cr_SVM(P, U, max_neg_ratio=0.1, noise_lvl=0.1, text=True)
print("\nTraining CR-SVM took %s seconds\n" % (time.time() - start_time))

def sort_cr_svm(sentences):
    return sorted(zip(crsvm.predict_proba(sentences),
                      sentences),
                  key=lambda x: x[0][1],
                  reverse=True)

def top_bot_12_cr_svm(predictions, name):
    print("\ncr-svm", name, "prediction", sum([1 for x in predictions if x[0][1] > 0.5]), "/",
          num_rows(predictions))
    print(name, "top-12")
    [print(x) for x in (predictions[0:12])]
    print(name, "bot-12")
    [print(x) for x in (predictions[-12:])]


civ_lab_sim = sort_cr_svm(civic)
top_bot_12_cr_svm(civ_lab_sim, "civic")

abs_lab_sim = sort_cr_svm(abstracts)
top_bot_12_cr_svm(abs_lab_sim, "abstracts")

oth_lab_sim = sort_cr_svm(piboso_other)
top_bot_12_cr_svm(oth_lab_sim, "piboso-other")

out_lab_sim = sort_cr_svm(piboso_outcome)
top_bot_12_cr_svm(out_lab_sim, "piboso-outcome")

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
# sys.exit(0)
# ----------------------------------------------------------------

def predicted_df(sentences):
    return pd.DataFrame(data={"Label" : model.predict(sentences),
                              "Text"  : sentences,
                              "Tokens": preppy.transform(sentences)},
                        columns=["Label", "Text", "Tokens"])


preppy = transformers.BasicPreprocessor()

lab_civ = predicted_df(civic)
lab_abs = predicted_df(abstracts)
lab_oth = predicted_df(piboso_other)
lab_out = predicted_df(piboso_outcome)

# save sentences with predicted labels to csv

print("civic: prediction", sum(lab_civ["Label"].values), "/", len(civic))
print("abstracts: prediction", sum(lab_abs["Label"].values), "/", len(abstracts))
print("piboso other: prediction", sum(lab_oth["Label"].values), "/", len(piboso_other))
print("piboso outcome: prediction", sum(lab_out["Label"].values), "/", len(piboso_outcome))

lab_civ.to_csv("./labelled_i-em_civic.csv")
lab_abs.to_csv("./labelled_i-em_abstracts.csv")
lab_oth.to_csv("./labelled_i-em_other.csv")
lab_out.to_csv("./labelled_i-em_outcome.csv")

yhat3 = model.predict(loaders.sentences_piboso_pop_bg_oth())
print("piboso pop bg oth: prediction", sum(yhat3), "/", len(yhat3))
