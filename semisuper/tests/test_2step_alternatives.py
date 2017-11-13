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

P = random.sample(P, 2000)
U = random.sample(U, 2000)

# ------------------
# SPY-SVM Test

print("\n\n"
      "------------\n"
      "SPY-SVM TEST\n"
      "------------\n")

start_time = time.time()
spysvm = pu_two_step.spy_SVM(P, U, spy_ratio=0.15, max_neg_ratio=0.1, tolerance=0.15, noise_lvl=0.2, text=True)
print("\nTraining SPY-SVM took %s seconds\n" % (time.time() - start_time))


def sort_spy_svm(sentences):
    return sorted(zip(spysvm.predict_proba(sentences),
                      sentences),
                  key=lambda x: x[0][1],
                  reverse=True)


def top_bot_12_spy_svm(predictions, name):
    print("\nspy-svm", name, "prediction", sum([1 for x in predictions if x[0][1] > 0.5]), "/",
          num_rows(predictions))
    print(name, "top-12")
    [print(x) for x in (predictions[0:12])]
    print(name, "bot-12")
    [print(x) for x in (predictions[-12:])]


civ_lab_sim = sort_spy_svm(civic)
top_bot_12_spy_svm(civ_lab_sim, "civic")

abs_lab_sim = sort_spy_svm(abstracts)
top_bot_12_spy_svm(abs_lab_sim, "abstracts")

hocpos_lab_sim = sort_spy_svm(hocpos)
top_bot_12_spy_svm(hocpos_lab_sim, "HoC-pos")

hocneg_lab_sim = sort_spy_svm(hocpos)
top_bot_12_spy_svm(hocpos_lab_sim, "HoC-neg")

oth_lab_sim = sort_spy_svm(piboso_other)
top_bot_12_spy_svm(oth_lab_sim, "piboso-other")

out_lab_sim = sort_spy_svm(piboso_outcome)
top_bot_12_spy_svm(out_lab_sim, "piboso-outcome")


# ------------------
# ROC-SVM Test

print("\n\n"
      "------------\n"
      "ROC-EM TEST\n"
      "------------\n")

start_time = time.time()
rocem = pu_two_step.roc_EM(P, U, max_neg_ratio=0.1, text=True)
print("\nTraining ROC-EM took %s seconds\n" % (time.time() - start_time))


def sort_roc_em(sentences):
    return sorted(zip(rocem.predict_proba(sentences),
                      sentences),
                  key=lambda x: x[0][1],
                  reverse=True)


def top_bot_12_roc_em(predictions, name):
    print("\nroc-em", name, "prediction", sum([1 for x in predictions if x[0][1] > 0.5]), "/",
          num_rows(predictions))
    print(name, "top-12")
    [print(x) for x in (predictions[0:12])]
    print(name, "bot-12")
    [print(x) for x in (predictions[-12:])]


civ_lab_sim = sort_roc_em(civic)
top_bot_12_roc_em(civ_lab_sim, "civic")

abs_lab_sim = sort_roc_em(abstracts)
top_bot_12_roc_em(abs_lab_sim, "abstracts")

hocpos_lab_sim = sort_roc_em(hocpos)
top_bot_12_roc_em(hocpos_lab_sim, "HoC-pos")

hocneg_lab_sim = sort_roc_em(hocpos)
top_bot_12_roc_em(hocpos_lab_sim, "HoC-neg")

oth_lab_sim = sort_roc_em(piboso_other)
top_bot_12_roc_em(oth_lab_sim, "piboso-other")

out_lab_sim = sort_roc_em(piboso_outcome)
top_bot_12_roc_em(out_lab_sim, "piboso-outcome")
