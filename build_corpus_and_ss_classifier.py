from __future__ import absolute_import, division, print_function

from semisuper import ss_model_selection, cleanup_corpora
from semisuper.loaders import load_pipeline
from semisuper.helpers import num_rows
import pandas as pd
import pickle
from datetime import datetime
import os
import numpy as np
from semisuper.loaders import abstracts2pmid_pos_sentence_title


def train_pipeline(from_scratch=False, write=True, outpath=None, mode=None, ratio=1.0):
    if not from_scratch:
        try:
            return load_pipeline()
        except:
            pass

    print("Building new classifier")

    P, N, U = cleanup_corpora.clean_corpus_pnu(ratio=ratio, mode=mode)

    print("P (HoC labelled + CIViC)", num_rows(P),
          "\tN (HoC unlabelled)", num_rows(N),
          "\tU (CIViC source abstracts)", num_rows(U))

    best_pipeline = ss_model_selection.best_model_cross_val(P, N, U, fold=10)

    if write or outpath:
        outpath = outpath or file_path("./semisuper/pickles/semi_pipeline.pickle")
        print("Pickling pipeline to", outpath)
        with open(outpath, "wb") as f:
            pickle.dump(best_pipeline, f)

    return best_pipeline


def save_silver_standard(pipeline, write=True, outpath=None):
    print("Building new silver standard")

    float_format = '%.4g'

    pmid, pos, text, title = [0, 1, 2, 3]

    abstracts = np.array(abstracts2pmid_pos_sentence_title())

    if hasattr(pipeline, 'decision_function'):
        dec_fn = pipeline.decision_function(abstracts[:, text])
    elif hasattr(pipeline, 'predict_proba'):
        dec_fn = np.abs(pipeline.predict_proba(abstracts[:, text])[:, 1])
    else:
        dec_fn = [0] * num_rows(abstracts)

    y = pipeline.predict(abstracts[:, text]).astype(int)

    abs_classified = pd.DataFrame(data={"label"            : y,
                                        "decision_function": [float_format % df for df in dec_fn],
                                        "pmid"             : abstracts[:, pmid],
                                        "sentence_pos"     : [float_format % float(pos) for pos in abstracts[:, pos]],
                                        "text"             : abstracts[:, text],
                                        "title"            : abstracts[:, title]
                                        },
                                  columns=["label", "decision_function", "pmid", "sentence_pos", "text", "title"])

    if write or outpath:
        outpath = outpath or file_path("./semisuper/silver_standard/silver_standard.tsv")
        print("Writing silver standard corpus to", outpath)
        abs_classified.to_csv(outpath, sep="\t", float_format=float_format)

    return abs_classified


def train_build(from_scratch=True, mode=None, ratio=1.0):
    now = datetime.now().strftime('%Y-%m-%d_%H-%M')

    pipeline = train_pipeline(from_scratch=from_scratch,
                              mode=mode,
                              ratio=ratio,
                              write=True,
                              outpath=None
                              # file_path("./semisuper/pickles/semi_pipeline" + now + '.pickle')
                              )

    silver_standard = save_silver_standard(pipeline,
                                           write=True,
                                           outpath=None
                                           # file_path("./semisuper/silver_standard/silver_standard" + now + '.tsv')
                                           )

    return silver_standard


# ----------------------------------------------------------------

def file_path(file_relative):
    """return the correct file path given the file's path relative to helpers"""
    return os.path.join(os.path.dirname(__file__), file_relative)


# ----------------------------------------------------------------
# Execution
# ----------------------------------------------------------------

if __name__ == "__main__":
    train_build(ratio=1.0)
