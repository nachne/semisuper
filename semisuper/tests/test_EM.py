from semisuper import transformers, loaders, pu_two_step
from semisuper.helpers import num_rows
import random
import pandas as pd
import time
import os.path

civic, abstracts = loaders.sentences_civic_abstracts()

print("CIViC sentences:", len(civic))
print("Abstract sentences:", len(abstracts))

piboso_other = loaders.sentences_piboso_other()
piboso_outcome = loaders.sentences_piboso_outcome()

print("PIBOSO sentences:", len(piboso_other))

P = civic
U = abstracts

P = random.sample(civic, 4000) + random.sample(piboso_outcome, 0)
U = random.sample(abstracts, 4000) + random.sample(P, 0)


def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)


# ------------------
# S-EM-Test

print("\n\n"
      "S-EM TEST\n"
      "---------\n")

start_time = time.time()

model = pu_two_step.s_EM(P, U, spy_ratio=0.1, tolerance=0.1, text=True)

print("\nS-EM took %s seconds\n" % (time.time() - start_time))

# ------------------
# I-EM-Test

print("\n\n"
      "I-EM TEST\n"
      "---------\n")

start_time = time.time()

model = pu_two_step.i_EM(P, U, max_pos_ratio=0.5, max_imbalance=1.0, tolerance=0.15, text=True)

print("\nEM took %s seconds\n" % (time.time() - start_time))


# print(dummy_pipeline.show_most_informative_features(model))

# ----------------------------------------------------------------
# sys.exit(0)
# ----------------------------------------------------------------

def predicted_df(sentences):
    return pd.DataFrame(data={"Label" : model.predict(sentences),
                              "Text"  : sentences,
                              "Tokens": preppy.transform(sentences)},
                        columns=["Label", "Text", "Tokens"])


preppy = transformers.BasicPreprocessor()

lab_civ = predicted_df(civic)
lab_abs = predicted_df(abstracts)
lab_oth = predicted_df(piboso_other)
lab_out = predicted_df(piboso_outcome)

# save sentences with predicted labels to csv

print("civic: prediction", sum(lab_civ["Label"].values), "/", len(civic))
print("abstracts: prediction", sum(lab_abs["Label"].values), "/", len(abstracts))
print("piboso other: prediction", sum(lab_oth["Label"].values), "/", len(piboso_other))
print("piboso outcome: prediction", sum(lab_out["Label"].values), "/", len(piboso_outcome))

lab_civ.to_csv(file_path("../output/labelled_i-em_civic.csv"))
lab_abs.to_csv(file_path("../output/labelled_i-em_abstracts.csv"))
lab_oth.to_csv(file_path("../output/labelled_i-em_other.csv"))
lab_out.to_csv(file_path("../output/labelled_i-em_outcome.csv"))

yhat3 = model.predict(loaders.sentences_piboso_pop_bg_oth())
print("piboso pop bg oth: prediction", sum(yhat3), "/", len(yhat3))
