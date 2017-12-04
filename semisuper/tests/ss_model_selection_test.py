from semisuper import ss_model_selection, cleanup_sources
from semisuper.helpers import num_rows

P, N, U, X_test, y_test = cleanup_sources.clean_corpus_pnu(ratio=1.0)

print("Training sets: \tP (HoC labelled + CIViC)", num_rows(P),
      "\tN (HoC unlabelled)", num_rows(N),
      "\tU (CIViC source abstracts)", num_rows(U),
      "\tTest (HoC labelled + CIViC VS. HoC unlabelled):", num_rows(X_test))

ss_model_selection.getBestModel(P, N, U, X_test, y_test)
