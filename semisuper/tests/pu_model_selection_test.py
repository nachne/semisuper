from semisuper import loaders, pu_two_step, pu_biased_svm, basic_pipeline, pu_model_selection, cleanup_sources
from semisuper.helpers import num_rows, densify
import random
from sklearn.model_selection import train_test_split


P, U, X_test, y_test = cleanup_sources.clean_corpus_pu(ratio=0.2)

print("Training sets: \tP (HoC labelled + CIViC)", num_rows(P),
      "\tU (HoC unlabelled + CIViC source abstracts)", num_rows(U),
      "\tTest (HoC labelled VS. HoC unlabelled):", num_rows(X_test))

pu_model_selection.getBestModel(P, U, X_test, y_test)
