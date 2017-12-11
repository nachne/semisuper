from semisuper import ss_model_selection, cleanup_sources
from semisuper.helpers import num_rows
import pandas as pd
import pickle
from datetime import datetime
import os
import sys
import numpy as np
from semisuper.loaders import abstract_pmid_pos_sentences


def train_pipeline(from_scratch=False, write=True, outpath=None, ratio=1.0):
    if not from_scratch:
        try:
            with open(file_path("../pickles/civic_pipeline.pickle"), "rb") as f:
                best_pipeline = pickle.load(f)
            return best_pipeline
        except:
            pass

    P, N, U = cleanup_sources.clean_corpus_pnu(ratio=ratio)

    print("P (HoC labelled + CIViC)", num_rows(P),
          "\tN (HoC unlabelled)", num_rows(N),
          "\tU (CIViC source abstracts)", num_rows(U))

    best_pipeline = ss_model_selection.best_model_cross_val(P, N, U, fold=5)

    if write or outpath:
        outpath = outpath or file_path("../pickles/civic_pipeline.pickle")
        print("Pickling pipeline to", outpath)
        with open(outpath, "wb") as f:
            pickle.dump(best_pipeline, f)

    return best_pipeline


def save_silver_standard(pipeline, write=True, outpath=None):
    abstracts = np.array(abstract_pmid_pos_sentences())
    y = pipeline.predict(abstracts[:, 2]).astype(int)

    if hasattr(pipeline, 'decision_function'):
        dec_fn = pipeline.decision_function(abstracts[:, 2])
    elif hasattr(pipeline, 'predict_proba'):
        dec_fn = np.abs(pipeline.predict_proba(abstracts[:, 2])[:, 1])
    else:
        dec_fn = [-999] * num_rows(abstracts)

    abs_classified = pd.DataFrame(data={"label"            : y,
                                        "decision_function": dec_fn,
                                        "pmid"             : abstracts[:, 0],
                                        "sentence_pos"     : abstracts[:, 1],
                                        "text"             : abstracts[:, 2],
                                        },
                                  columns=["label", "decision_function", "pmid", "sentence_pos", "text"])

    if write or outpath:
        outpath = outpath or file_path("../output/silver_standard.tsv")
        print("Writing silver standard corpus to", outpath)
        abs_classified.to_csv(outpath, sep="\t")

    return abs_classified


def file_path(file_relative):
    """return the correct file path given the file's path relative to helpers"""
    return os.path.join(os.path.dirname(__file__), file_relative)


def main(args):
    pipeline = train_pipeline(from_scratch=True, outpath=file_path("../pickles/civic_pipeline" +
                                                                   datetime.now().strftime('%Y-%m-%d_%H:%M.pickle')),
                              ratio=1.0)

    silver_standard = save_silver_standard(pipeline, outpath=file_path("../output/silver_standard" +
                                                                       datetime.now().strftime('%Y-%m-%d_%H:%M.tsv')))

    print(silver_standard)

    return


if __name__ == "__main__":
    main(sys.argv[1:])
