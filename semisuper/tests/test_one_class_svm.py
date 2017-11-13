from semisuper import transformers, basic_pipeline, loaders
from semisuper.pu_one_class_svm import one_class_svm
from semisuper.pu_ranking import ranking_cos_sim
import random
from operator import itemgetter
import pandas as pd


civic, abstracts = loaders.sentences_civic_abstracts()

print("CIViC sentences:", len(civic))
print("abstract sentences:", len(abstracts))

piboso_other = loaders.sentences_piboso_other()
piboso_outcome = loaders.sentences_piboso_outcome()

print("PIBOSO sentences:", len(piboso_other))

pos = random.sample(civic, 2000) + random.sample(piboso_outcome, 0)
neg = random.sample(abstracts, 2000) + random.sample(piboso_other, 0)

X = pos + neg
y = ["pos"] * len(pos) + ["neg"] * len(neg)

# ------------------
# one-class SVM Test

print("\n\n"
      "------------------\n"
      "ONE-CLASS SVM TEST\n"
      "------------------\n")

one_class = one_class_svm(civic, kernel="rbf")

civ_lab_1cl = sorted(zip(one_class.predict_proba(civic), civic), key=itemgetter(0), reverse=True)
print("1-class svm civic prediction", sum([x for x in civ_lab_1cl if x > 0]), "/", len(civ_lab_1cl))
print("\ncivic top-12")
[print(x) for x in (civ_lab_1cl[0:12])]
print("civic bot-12")
[print(x) for x in (civ_lab_1cl[-12:])]

abs_lab_1cl = sorted(zip(one_class.predict_proba(abstracts), abstracts), key=itemgetter(0), reverse=True)
print("1-class svm abstracts prediction", sum([x for x in abs_lab_1cl if x > 0]), "/", len(abs_lab_1cl))

print("\nabstracts top-12")
[print(x) for x in (abs_lab_1cl[0:12])]
print("abstracts bot-12")
[print(x) for x in (abs_lab_1cl[-12:])]

oth_lab_1cl = sorted(zip(one_class.predict_proba(piboso_other), piboso_other), key=itemgetter(0), reverse=True)
print("1-class svm piboso-other prediction", sum([x for x in oth_lab_1cl if x > 0]), "/", len(oth_lab_1cl))
print("\npiboso other top-12")
[print(x) for x in (oth_lab_1cl[0:12])]
print("piboso other bot-12")
[print(x) for x in (oth_lab_1cl[-12:])]

out_lab_1cl = sorted(zip(one_class.predict_proba(piboso_outcome), piboso_outcome), key=itemgetter(0), reverse=True)
print("1-class svm piboso-outcome prediction", sum([x for x in out_lab_1cl if x > 0]), "/", len(out_lab_1cl))
print("\npiboso outcome top-12")
[print(x) for x in (out_lab_1cl[0:12])]
print("piboso outcome bot-12")
[print(x) for x in (out_lab_1cl[-12:])]
print("\n")
