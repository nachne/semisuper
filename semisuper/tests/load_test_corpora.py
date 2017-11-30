import glob
import os.path
import re

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
import numpy as np


# ----------------------------------------------------------------
# top-level for getting noisy training and validation data
# ----------------------------------------------------------------

# ------------------------------
# for PU

def P_U_p_n_uci_sentences(neg_noise=0.05, pos_in_u=0.6, test_size=0.1):
    P, N = pos_neg_sentences()
    return mixed_pu(P, N, neg_noise=neg_noise, pos_in_u=pos_in_u, test_size=test_size)


def P_U_p_n_amazon(neg_noise=0.05, pos_in_u=0.6, test_size=0.1):
    P, N = pos_neg_amazon()
    return mixed_pu(P, N, neg_noise=neg_noise, pos_in_u=pos_in_u, test_size=test_size)


def P_U_p_n_sms_spam(neg_noise=0.05, pos_in_u=0.6, test_size=0.1):
    P, N = pos_neg_sms_spam()
    return mixed_pu(P, N, neg_noise=neg_noise, pos_in_u=pos_in_u, test_size=test_size)


def list_P_U_p_n_20_newsgroups(neg_noise=0.05, pos_in_u=0.6, test_size=0.1, ratio=0.5):
    all_categories = pos_neg_20_newsgroups(ratio=ratio)

    for (P, N) in all_categories:
        yield mixed_pu(P, N, neg_noise=neg_noise, pos_in_u=pos_in_u, test_size=test_size)


# ------------------------------
# for Semi-Supervised

def P_N_U_p_n_uci_sentences(neg_noise=0.05, pos_in_u=0.6, neg_in_u=0.6, test_size=0.1):
    P, N = pos_neg_sentences()
    return mixed_pnu(P, N, neg_noise=neg_noise, pos_in_u=pos_in_u, neg_in_u=neg_in_u, test_size=test_size)


def P_N_U_p_n_amazon(neg_noise=0.05, pos_in_u=0.6, neg_in_u=0.6, test_size=0.1):
    P, N = pos_neg_amazon()
    return mixed_pnu(P, N, neg_noise=neg_noise, pos_in_u=pos_in_u, neg_in_u=neg_in_u, test_size=test_size)


def P_N_U_p_n_sms_spam(neg_noise=0.05, pos_in_u=0.6, neg_in_u=0.6, test_size=0.1):
    P, N = pos_neg_sms_spam()
    return mixed_pnu(P, N, neg_noise=neg_noise, pos_in_u=pos_in_u, neg_in_u=neg_in_u, test_size=test_size)


def list_P_N_U_p_n_20_newsgroups(neg_noise=0.05, pos_in_u=0.6, neg_in_u=0.6, test_size=0.1, ratio=0.5):
    all_categories = pos_neg_20_newsgroups(ratio=ratio)

    for (P, N) in all_categories:
        yield mixed_pnu(P, N, neg_noise=neg_noise, pos_in_u=pos_in_u, neg_in_u=neg_in_u, test_size=test_size)


# ----------------------------------------------------------------
# loader functions
# ----------------------------------------------------------------

def pos_neg_20_newsgroups(ratio=0.5):
    ng = fetch_20newsgroups(subset="all", remove=("headers",))

    X = np.array(ng.data)
    y = np.array(ng.target)

    if ratio < 1.0:
        X, _, y, _ = train_test_split(X, y, train_size=ratio)

    supercat_indices = [[1, 2, 3, 4, 5], [7, 8, 9, 10], [11, 12, 13, 14], [16, 17, 18, 19],
                       # [0], [6], [15],
                       ]
    supercat_names = ["COMP", "REC", "SCI","TALK", "ALT", "MISC", "SOC"]
    for i in range(len(supercat_indices)):
        print("CATEGORY:", supercat_names[i])
        P = X[np.where(np.isin(y, supercat_indices[i]))]
        N = X[np.where(np.isin(y, supercat_indices[i], invert=True))]
        yield P, N


def pos_neg_sentences():
    def file_path(file_relative):
        """return the correct file path given the file's path relative to helpers"""
        return os.path.join(os.path.dirname(__file__), file_relative)

    positive = []
    negative = []

    label_re = re.compile("^\w+\t")

    # print(len(glob.glob(file_path("../resources/uci_sentence_corpus/labeled_articles/*.txt"))))

    for filename in glob.glob(file_path(file_path("../resources/corpora/uci_sentence_corpus/labeled_articles/" +
                                                          '*.txt'))):
        with open(filename, 'r') as f:

            lines = f.read().split('\n')

            for line in lines[1:]:
                if not len(line) or line[0] == '#' or not label_re.findall(line):
                    continue

                label = label_re.findall(line)[0]
                # print("label:", label)

                sentence = label_re.sub("", line)
                # print("sentence:", sentence)

                if label == 'OWNX\t' or label == 'AIMX\t' or label == 'CONT\t':
                    positive.append(sentence)
                else:
                    negative.append(sentence)

    return positive, negative


def pos_neg_amazon():
    p_amaz, n_amaz = pos_neg_single_file(path="../resources/corpora/sentiment_labelled_sentences/"
                                              "amazon_cells_labelled.txt",
                                         label_re_str="\t[01]",
                                         pos_str="\t1")

    p_yelp, n_yelp = pos_neg_single_file(path="../resources/corpora/sentiment_labelled_sentences/"
                                              "yelp_labelled.txt",
                                         label_re_str="\t[01]",
                                         pos_str="\t1")

    p_imdb, n_imdb = pos_neg_single_file(path="../resources/corpora/sentiment_labelled_sentences/"
                                              "imdb_labelled.txt",
                                         label_re_str="\t[01]",
                                         pos_str="\t1")

    return (p_amaz + p_imdb + p_yelp,
            n_amaz + n_imdb + n_yelp)


def pos_neg_sms_spam():
    return pos_neg_single_file(path="../resources/corpora/smsspamcollection/"
                                    "SMSSpamCollection",
                               label_re_str="\w+\t",
                               pos_str="spam\t")


def pos_neg_single_file(path, label_re_str, pos_str):
    def file_path(file_relative):
        """return the correct file path given the file's path relative to helpers"""
        return os.path.join(os.path.dirname(__file__), file_relative)

    positive = []
    negative = []

    label_re = re.compile(label_re_str)

    # print(len(glob.glob(file_path("../resources/uci_sentence_corpus/labeled_articles/*.txt"))))

    filename = file_path(file_path(path))
    with open(filename, 'r') as f:

        lines = f.read().split('\n')

        for line in lines[1:]:
            if not len(line) or line[0] == '#' or not label_re.findall(line):
                continue

            label = label_re.findall(line)[0]
            # print("label:", label)

            sentence = label_re.sub("", line)
            # print("sentence:", sentence)

            if label == pos_str:
                positive.append(sentence)
            else:
                negative.append(sentence)

    return positive, negative


# ----------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------

def mixed_pu(P, N, neg_noise=0.05, pos_in_u=0.6, test_size=0.2):
    """returns P with optional negative noise, U with pos_in_u of P, and positive and negative validation set"""

    print("\nParameters for training data:\n",
          100 * pos_in_u, "% of positive documents are hidden in unlabelled set U.\n",
          100 * neg_noise, "% of P is actually negative, to simulate noise.\n")

    P_train, P_test = train_test_split(P, test_size=test_size)
    N_train, N_test = train_test_split(N, test_size=test_size)
    P_P, P_U = train_test_split(P_train, test_size=pos_in_u)
    N_U, N_noise = train_test_split(N_train, test_size=int(neg_noise * ((1+neg_noise) * len(P_P))))

    P_ = np.concatenate((P_P, N_noise))
    U_ = np.concatenate((P_U, N_U))

    return P_, U_, np.array(P_test), np.array(N_test)


def mixed_pnu(P, N, neg_noise=0.05, pos_in_u=0.6, neg_in_u=0.6, test_size=0.2):
    """returns P with optional negative noise, U with pos_in_u of P, and positive and negative validation set"""

    print("\nParameters for training data:\n",
          100 * pos_in_u, "% of positive documents are hidden in unlabelled set U.\n",
          100 * neg_in_u, "% of negative documents are hidden in unlabelled set U.\n",
          100 * neg_noise, "% of P is actually negative, to simulate noise.\n")

    P_train, P_test = train_test_split(P, test_size=test_size)
    N_train, N_test = train_test_split(N, test_size=test_size)
    P_P, P_U = train_test_split(P_train, test_size=pos_in_u)
    N_N, N_U = train_test_split(N_train, test_size=neg_in_u)
    N_N, N_noise = train_test_split(N_N, test_size=int(neg_noise * ((1+neg_noise) * len(P_P))))

    P_ = np.concatenate((P_P, N_noise))
    U_ = np.concatenate((P_U, N_U))

    return P_, np.array(N_N), U_, np.array(P_test), np.array(N_test)
