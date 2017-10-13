import pickle
import multiprocessing as multi
import pandas as pd
from nltk.corpus import wordnet, stopwords


# concept mapping

class WordReplacer(object):
    def __init__(self, word_map):
        self.word_map = word_map

    def replace(self, word):
        return self.word_map.get(word, word)

    def replace_all(self, words):
        return [self.replace(w) for w in words]


class HypernymMapper(WordReplacer):
    def __init__(self):
        dict = load_hypernyms()
        super(HypernymMapper, self).__init__(dict)


def load_hypernyms():
    """read hypernym dict from disk or build from tsv files"""
    try:
        with open("hypernyms.pickle", "rb") as f:
            hypernyms = pickle.load(f)
            print("loaded hypernyms from disk.")
    except:
        print("building hypernym resources...")
        hypernyms = build_hypernym_dict()
        with open("hypernyms.pickle", "wb") as f:
            pickle.dump(hypernyms, f)
            print("built hypernym dict and wrote to disk.")
    return hypernyms


def build_hypernym_dict():
    concepts = ["chemical", "disease", "gene", "mutation"]

    with multi.Pool(len(concepts)) as p:
        dicts = list(p.map(make_hypernym_entries, concepts))

    dict = dicts[0].copy()
    for d in dicts[1:]:
        dict.update(d)

    return (dict)


def make_hypernym_entries(h):
    entries = {}
    source = pd.read_csv("./" + h + "2pubtator", sep='\t', dtype=str)

    common_words = set(stopwords.words('english'))

    for line in source["Mentions"]:
        for word in str(line).split("|"):
            # only include single words not appearing in normal language
            if not (" " in word
                    or word in common_words
                    or word in entries
                    or wordnet.synsets(word)):
                entries[word] = "_" + h + "_"

    return entries
