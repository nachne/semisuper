from semisuper import ss_model_selection, cleanup_sources
from semisuper.helpers import num_rows, flatten
import pandas as pd
import numpy as np
from loaders import abstract_pmid_pos_sentences
from sklearn.pipeline import Pipeline

P, N, U = cleanup_sources.clean_corpus_pnu(ratio=1.0)

print("P (HoC labelled + CIViC)", num_rows(P),
      "\tN (HoC unlabelled)", num_rows(N),
      "\tU (CIViC source abstracts)", num_rows(U))

best_pipeline = ss_model_selection.best_model_cross_val(P, N, U)

abstracts = np.array(abstract_pmid_pos_sentences())
y = best_pipeline.predict(abstracts[:,2])
conf = best_pipeline.decision_function(abstracts[:,2])

abs_clfd = list(zip(y, conf, abstracts[:,1], abstracts[:,2]))

for i in range(400,800):
      print(abs_clfd[i])
