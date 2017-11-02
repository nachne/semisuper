from semisuper import loaders, pu_biased_svm
from numpy import arange
import random
import time

civic, abstracts = loaders.sentences_civic_abstracts()

print("CIViC sentences:", len(civic))
print("Abstract sentences:", len(abstracts))

piboso_other = loaders.sentences_piboso_other()
piboso_outcome = loaders.sentences_piboso_outcome()

print("PIBOSO sentences:", len(piboso_other))

P = civic
U = abstracts

P = random.sample(civic, 2000) + random.sample(piboso_outcome, 0)
U = random.sample(abstracts, 2000) + random.sample(P, 0)

# ------------------

print("\n\n"
      "---------------\n"
      "BIASED-SVM TEST\n"
      "---------------\n")

start_time = time.time()
biased_svm = pu_biased_svm.biased_SVM_weight_selection(P, U,
                                                       # Cs=[10 ** x for x in range(-4, 5, 4)],
                                                       Cs_neg=arange(0.01, 0.63, 0.32),
                                                       Cs_pos_factors=range(1, 2200, 200),
                                                       text=True)
print("\nTraining Biased-SVM took %s seconds\n" % (time.time() - start_time))


def sort_biased_svm(sentences):
    return sorted(zip(biased_svm.predict(sentences),
                      biased_svm.predict_proba(sentences),
                      sentences),
                  key=lambda x: x[1][1],
                  reverse=True)


def top_bot_12_biased_svm(predictions, name):
    print("\nbiased-svm", name, "prediction", sum([1 for x in predictions if x[0] == 1]), "/",
          len(predictions))
    print(name, "top-12")
    [print(x) for x in (predictions[0:12])]
    print(name, "bot-12")
    [print(x) for x in (predictions[-12:])]


sentences = ["I am an automobile",
             "we found a hexagonal arthritis alien in my ankle",
             "imatinib p=14 mutation progression-free survival",
             "cancer carcinoma 13 11 0.0 1<4"]
print("Nonsense Sentences, expecting [0, ?, 1, 1]")
print(biased_svm.predict(sentences))
print(biased_svm.predict_proba(sentences))

civ_lab_sim = sort_biased_svm(civic)
top_bot_12_biased_svm(civ_lab_sim, "civic")

abs_lab_sim = sort_biased_svm(abstracts)
top_bot_12_biased_svm(abs_lab_sim, "abstracts")

oth_lab_sim = sort_biased_svm(piboso_other)
top_bot_12_biased_svm(oth_lab_sim, "piboso-other")

out_lab_sim = sort_biased_svm(piboso_outcome)
top_bot_12_biased_svm(out_lab_sim, "piboso-outcome")
