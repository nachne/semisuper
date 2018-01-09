import pickle
from geniatagger import GeniaTagger
import multiprocessing as multi
from semisuper import loaders
from semisuper.helpers import flatten

taggers = [GeniaTagger("/Users/emperor/HU/BA/semisuper/semisuper/resources/geniatagger-3.0.2/geniatagger")
           for _ in range(4)]


def representation(sentences, tagger=None):
    return [tagger.parse(s) for s in sentences]


def repr(x):
    return representation(x[0], taggers[x[1]])

print(repr([flatten(loaders.sentences_HoC()[:1000]), 0]))

with multi.Pool(4) as p:
    tagged = list(p.map(repr, zip([flatten(loaders.sentences_HoC()[:1000]),
                                   flatten(loaders.sentences_HoC()[:1000]),
                                   flatten(loaders.sentences_HoC()[:1000]),
                                   flatten(loaders.sentences_HoC()[:1000]),
                                   # flatten(loaders.sentences_civic_abstracts()[0][:1000]),
                                   # flatten(loaders.sentences_civic_abstracts()[1]),
                                   # flatten(loaders.sentences_piboso_outcome())
                                   ],
                                  [0, 1, 2, 3])))

print(tagged)

# ----------------------------------------------------------------


tagger = None

with open("/Users/emperor/HU/BA/semisuper/semisuper/pickles/semi_pipeline.pickle", "rb") as f:
    ppl = pickle.load(f)

for x in [["Mutations highly significant", "Janus kinase-2 tyrosine kinase mutation", "(c) 2015 AACR"],
          ["Mutations insignificant", "Janus kinase-2 tyrosine kinase mutation JAK-2", "(c) 2015 AACR"],
          ["oioioi", "aiaiai", "therefore we must resort to inhibiting all kinds of tumours and malignancies"],
          ["there is no evidence", "why though, I ask?", "genomic mice are conquering Earth"]]:
    print(ppl.decision_function(x))

print(tagger)
