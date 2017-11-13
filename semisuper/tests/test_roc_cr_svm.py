from semisuper import transformers, loaders, pu_two_step
from semisuper.helpers import num_rows
import random
import pandas as pd
import time

civic, abstracts = loaders.sentences_civic_abstracts()
hocpos, hocneg = loaders.sentences_HoC()

print("CIViC sentences:", len(civic))
print("Abstract sentences:", len(abstracts))
print("HoC positive sentences:", len(hocpos))
print("HoC negative sentences:", len(hocneg))

piboso_other = loaders.sentences_piboso_other()
piboso_outcome = loaders.sentences_piboso_outcome()

print("PIBOSO sentences:", len(piboso_other))

P = civic # + hocpos
U = abstracts # + hocneg

P = random.sample(P, 4000)
U = random.sample(U, 8000)

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

hocpos_lab_sim = sort_roc_svm(hocpos)
top_bot_12_cr_svm(hocpos_lab_sim, "HoC-pos")

hocneg_lab_sim = sort_roc_svm(hocpos)
top_bot_12_cr_svm(hocpos_lab_sim, "HoC-neg")

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
crsvm = pu_two_step.cr_SVM(P, U, max_neg_ratio=0.1, noise_lvl=0.5, text=True)
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

hocpos_lab_sim = sort_cr_svm(hocpos)
top_bot_12_cr_svm(hocpos_lab_sim, "HoC-pos")

hocneg_lab_sim = sort_cr_svm(hocneg)
top_bot_12_cr_svm(hocneg_lab_sim, "HoC-neg")

oth_lab_sim = sort_cr_svm(piboso_other)
top_bot_12_cr_svm(oth_lab_sim, "piboso-other")

out_lab_sim = sort_cr_svm(piboso_outcome)
top_bot_12_cr_svm(out_lab_sim, "piboso-outcome")
