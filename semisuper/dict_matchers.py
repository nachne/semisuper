import pickle
import multiprocessing as multi
import pandas as pd
import re
from nltk.corpus import wordnet, stopwords
import os.path


# concept mapping

class DictReplacer(object):
    def __init__(self, word_map):
        self.word_map = word_map

    def replace(self, word):
        return self.word_map.get(word, word)

    def replace_all(self, words):
        return [self.replace(w) for w in words]


class HypernymMapper(DictReplacer):
    def __init__(self):
        dictionary = self.load_hypernyms()
        super(HypernymMapper, self).__init__(dictionary)

    def load_hypernyms(self):
        """read hypernym dict from disk or build from tsv files"""
        try:
            with open(file_path("./pickles/hypernyms.pickle"), "rb") as f:
                hypernyms = pickle.load(f)
                # print("Loaded hypernyms from disk.")
        except IOError:
            print("Building hypernym resources...")
            hypernyms = self.build_hypernym_dict()
            with open(file_path("./pickles/hypernyms.pickle"), "wb") as f:
                pickle.dump(hypernyms, f)
                print("Built hypernym dict and wrote to disk.")
        return hypernyms

    def build_hypernym_dict(self):
        concepts = ["chemical", "disease", "gene", "mutation"]

        with multi.Pool(min(multi.cpu_count(), len(concepts))) as p:
            dicts = list(p.map(self.make_hypernym_entries, concepts))

        dictionary = dicts[0].copy()
        for d in dicts[1:]:
            dictionary.update(d)

        return dictionary

    def make_hypernym_entries(self, hypernym):
        entries = {}
        source = pd.read_csv(file_path("./resources/" + hypernym + "2pubtator.csv"),
                             sep='\t', dtype=str)

        with open(file_path("./resources/common_words.txt"), "r") as cw:
            common_words = set(cw.read().split("\n") + stopwords.words('english'))

        # only single words and no URLs etc
        illegal_substrs = re.compile("\s|\\\\|\.gov|\.org|\.com|http|www|")

        for line in source["Mentions"]:
            for word in str(line).split("|"):
                # only include single words not appearing in normal language
                if not (illegal_substrs.findall(word)
                        or word in common_words
                        or word in entries
                        or wordnet.synsets(word)):
                    entries[word] = "_" + hypernym + "_"

        return entries


# ----------------------------------------------------------------
# general helpers
# ----------------------------------------------------------------


def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)
