from semisuper import pu_model_selection, cleanup_corpora, loaders
from semisuper.helpers import num_rows
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from semisuper.loaders import abstracts2pmid_pos_sentence_title
import numpy as np
import random

civic, abstracts = loaders.sentences_civic_abstracts()
hocpos, hocneg = loaders.sentences_HoC()
piboso_other = loaders.sentences_piboso_other()
piboso_outcome = loaders.sentences_piboso_outcome()

print("CIViC sentences:", len(civic))
print("Abstract sentences:", len(abstracts))
print("HoC positive sentences:", len(hocpos))
print("HoC negative sentences:", len(hocneg))
# print("PIBOSO outcome sentences:", len(piboso_outcome))
# print("PIBOSO other sentences:", len(piboso_other))

hocpos_train, hocpos_test = train_test_split(hocpos, test_size=0.3)
hocneg_train, hocneg_test = train_test_split(hocneg, test_size=0.3)
civic_train, civic_test = train_test_split(civic, test_size=0.3)
abstracts_train, abstracts_test = train_test_split(abstracts, test_size=0.03)

# TODO check why this doesn't kill multiprocessing but clean corpus does

# P = random.sample(hocpos_train + civic_train, 4000)
# U = random.sample(hocneg_train + abstracts_train, 8000)
# half_test_size = 1000
# X_test = random.sample(hocpos_test, half_test_size) + random.sample(hocneg_test, half_test_size)
# y_test = [1] * half_test_size + [0] * half_test_size

# P = hocpos_train + civic_train
# U = hocneg_train + abstracts_train
# X_test = hocpos_test + hocneg_test
# y_test = [1] * num_rows(hocpos_test) + [0] * num_rows(hocneg_test)

P, U, X_test, y_test = cleanup_corpora.clean_corpus_pu(ratio=1.0)

print("Training sets: \tP (HoC labelled + CIViC)", num_rows(P),
      "\tU (HoC unlabelled + CIViC source abstracts)", num_rows(U),
      "\tTest (HoC labelled + CIViC VS. HoC unlabelled):", num_rows(X_test))

best = pu_model_selection.getBestModel(P, U, X_test, y_test)

best_pipeline = Pipeline([('vectorizer', best['vectorizer']),
                          ('selector', best['selector']),
                          ('clf', best['model'])])

abstracts = np.array(abstracts2pmid_pos_sentence_title())
y = best_pipeline.predict(abstracts[:, 2])

if hasattr(best_pipeline, 'decision_function'):
    conf = best_pipeline.decision_function(abstracts[:, 2])
elif hasattr(best_pipeline, 'predict_proba'):
    conf = np.abs(best_pipeline.predict_proba(abstracts[:, 2])[:, 1])
else:
    conf = [0] * num_rows(abstracts)

abs_clfd = list(zip(y, conf, abstracts[:, 1], abstracts[:, 2]))

for i in range(0, 20):
    print(abs_clfd[i])
