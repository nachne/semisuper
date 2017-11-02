import glob
import os.path
import re
from sklearn.model_selection import train_test_split


def P_U_p_n_uci():
    P, N = pos_neg_uci()
    return mixed_pu(P, N)

def pos_neg_uci():

    def file_path(file_relative):
        """return the correct file path given the file's path relative to helpers"""
        return os.path.join(os.path.dirname(__file__), file_relative)

    positive = []
    negative = []

    label_re = re.compile("^\w+\t")

    # print(len(glob.glob(file_path("../resources/uci_sentence_corpus/labeled_articles/*.txt"))))

    for filename in glob.glob(file_path(file_path("../resources/uci_sentence_corpus/labeled_articles/" +
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

def mixed_pu(P, N, neg_noise=0.05, pos_in_u=0.6):
    """returns P with optional negative noise, U with pos_in_u of P, and positive and negative validation set"""
    P_train, P_test = train_test_split(P, test_size=0.2)
    N_train, N_test = train_test_split(N, test_size=0.2)
    P_P, P_U = train_test_split(P_train, test_size=pos_in_u)
    N_U, N_noise = train_test_split(N_train, test_size=neg_noise)

    P_ = P_P + N_noise
    U_ = P_U + N_U

    return P_, U_, P_test, N_test





