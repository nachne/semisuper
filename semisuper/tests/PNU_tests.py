import random
import time

import numpy as np
from semisuper import loaders, basic_pipeline, ss_techniques
from semisuper.helpers import num_rows, unsparsify

civic, abstracts = loaders.sentences_civic_abstracts()
hocpos, hocneg = loaders.sentences_HoC()
piboso_other = loaders.sentences_piboso_other()
piboso_outcome = loaders.sentences_piboso_outcome()

print("CIViC sentences:", len(civic))
print("Abstract sentences:", len(abstracts))
print("HoC positive sentences:", len(hocpos))
print("HoC negative sentences:", len(hocneg))
print("PIBOSO outcome sentences:", len(piboso_outcome))
print("PIBOSO other sentences:", len(piboso_other))

# print("\n\nTRAINING ON CIVIC AND ABSTRACTS\n\n")

show_sentences = True

P_raw = random.sample(hocpos + civic, 1000)
N_raw = random.sample(hocneg, 1000)
U_raw = random.sample(abstracts, 2000)

print("\nTRAINING SEMI-SUPERVISED"
      "\tP: HOC POS"
      ", CIVIC"
      , "\tN: HOC NEG"
      , ",\tU: ABSTRACTS"
      )

words, wordgram_range = [False, (1, 3)]  # TODO change back to True, (1,3)
chars, chargram_range = [True, (2, 4)]  # TODO change back to True, (3,6)
rules, lemmatize = [True, True]

print("Fitting vectorizer")
vectorizer = basic_pipeline.vectorizer(words=words, wordgram_range=wordgram_range,
                                       chars=chars, chargram_range=chargram_range,
                                       rules=rules, lemmatize=lemmatize)
vectorizer.fit(np.concatenate((P_raw, U_raw)))

P = unsparsify(vectorizer.transform(P_raw))
N = unsparsify(vectorizer.transform(N_raw))
U = unsparsify(vectorizer.transform(U_raw))

print("P:", num_rows(P), "\tN:", num_rows(N), "\tU:", num_rows(U))
print("Features before selection:", np.shape(P)[1])

selector = basic_pipeline.selector()
selector.fit(np.concatenate((P, N, U)),
             np.concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N)), np.zeros(num_rows(U)))))
P = unsparsify(selector.transform(P))
N = unsparsify(selector.transform(N))
U = unsparsify(selector.transform(U))

print("Features after selection:", np.shape(P)[1])


# ------------------
# model testers
# ------------------

def test_all(P, N, U):
    test_label_propagation(P, N, U)
    # test_svm(P, N, U)
    # test_em(P, N, U)

    return


def test_label_propagation(P, N, U):
    print("\n\n"
          "---------\n"
          "LABEL PROPAGATION TEST\n"
          "---------\n")

    start_time = time.time()

    model = ss_techniques.propagate_labels(P, N, U)

    print("\nTraining Label Propagation took %s seconds\n" % (time.time() - start_time))

    print_sentences(model, "Label Propagation")
    return

def test_svm(P, N, U):
    print("\n\n"
          "---------\n"
          "SVM TEST\n"
          "---------\n")

    start_time = time.time()

    model = ss_techniques.iterate_SVM(P, N, U)

    print("\nIterating SVM took %s seconds\n" % (time.time() - start_time))

    print_sentences(model, "iterativeSVM")
    return


def test_em(P, N, U):
    print("\n\n"
          "---------\n"
          "EM TEST\n"
          "---------\n")

    start_time = time.time()

    model = ss_techniques.EM(P, N, U)

    print("\nTraining EM took %s seconds\n" % (time.time() - start_time))

    print_sentences(model, "EM")
    return


# ------------------
# helper to print amount of positive docs, top and bottom sentences per corpus
# ------------------

def print_sentences(model, modelname=""):
    if not show_sentences:
        return

    print("\n\n"
          "----------------\n"
          "{} SENTENCES\n"
          "----------------\n".format(modelname))

    def sort_model(sentences):
        sent_features = unsparsify(selector.transform(vectorizer.transform(sentences)))

        return sorted(zip(model.predict_proba(sent_features),
                          sentences),
                      key=lambda x: x[0][1],
                      reverse=True)

    def top_bot_12_model(predictions, name):
        pos = sum([1 for x in predictions if x[0][1] > x[0][0]])
        num = num_rows(predictions)

        print()
        print(modelname, name, "prediction", pos, "/", num, "(", pos / num, ")")
        print()
        print(modelname, name, "top 12 (sentences with highest ranking)\n")
        [print(x) for x in (predictions[0:12])]
        print()
        print(modelname, name, "bottom 12 (sentences with lowest ranking)\n")
        [print(x) for x in (predictions[-12:])]

    civ_labelled = sort_model(civic)
    top_bot_12_model(civ_labelled, "civic")

    abs_labelled = sort_model(abstracts)
    top_bot_12_model(abs_labelled, "abstracts")

    hocpos_labelled = sort_model(hocpos)
    top_bot_12_model(hocpos_labelled, "HoC-pos")

    hocneg_labelled = sort_model(hocneg)
    top_bot_12_model(hocneg_labelled, "HoC-neg")

    out_labelled = sort_model(piboso_outcome)
    top_bot_12_model(out_labelled, "piboso-outcome")

    oth_labelled = sort_model(piboso_other)
    top_bot_12_model(oth_labelled, "piboso-other")

    return


# ------------------

def print_params():
    print("words:", words, "\tword n-gram range:", wordgram_range,
          "\nchars:", chars, "\tchar n-gram range:", chargram_range,
          "\nrule-based preprocessing:", rules, "\tlemmatization:", lemmatize)
    return


# ------------------
# execute
# ------------------

print_params()
test_all(P, N, U)
