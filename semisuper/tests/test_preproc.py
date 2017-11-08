from semisuper import loaders
from semisuper.transformers import BasicPreprocessor, TextStats, FeatureNamePipeline
from semisuper.helpers import identity

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from numpy import concatenate

import pandas as pd
import random
import sys

# ----------------------------------------------------------------
# Data
# ----------------------------------------------------------------

civic, abstracts = loaders.sentences_civic_abstracts()

piboso_other = loaders.sentences_piboso_other()
piboso_outcome = loaders.sentences_piboso_outcome()

civic_ = random.sample(civic, 100)
abstracts_ = random.sample(abstracts, 100)

# ----------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------

pp = BasicPreprocessor()

pppl = Pipeline([
    ('preprocessor', pp),
    ('vectorizer', FeatureUnion(transformer_list=[("words", TfidfVectorizer(
            tokenizer=identity, preprocessor=None, lowercase=False, ngram_range=(1, 3))
                                                   ),
                                                  ("stats", FeatureNamePipeline([
                                                      ("stats", TextStats()),
                                                      ("vect", DictVectorizer())
                                                  ]))
                                                  ]
                                ))
])

# ----------------------------------------------------------------
# Regex tests
# ----------------------------------------------------------------

regex_concept_matches = ["p<=1 P>6",
                         "5-year-old y.o.",
                         "~1-2 23:24",
                         "a>=b f=1",
                         "2-fold ~1-fold",
                         "1:10 2-3",
                         "12.2% % 1%",
                         "1kg g mmol 2.2nm",
                         "rs2 rs333-wow",
                         "jan March february",
                         "1988 2001 20021",
                         "Zeroth first 2nd 22nd 23-th 9th",
                         "-1.0 .99 ~2",
                         "V.S. vS Versus I.E. ie. E.g. iv. Po p.o."]

[print(x, "\t", pp.transform([x])) for x in regex_concept_matches]

# ----------------------------------------------------------------
# abort

# sys.exit(0)

# ----------------------------------------------------------------
# Output
# ----------------------------------------------------------------

# pppl.fit(concatenate((civic_, abstracts_)))

print("\n----------------------------------------------------------------",
      "\nSome tests.",
      "\n----------------------------------------------------------------\n")

[print(pp.transform([x])) for x in ['CLINICALTRIALSGOV: NCT00818441.',
                                    'Eight inpatient rehabilitation facilities.',
                                    'There were 67 burst fractures, 48 compression fractures and 21 fracture dislocations, 8 flexion distraction fractures and 6 flexion rotation injuries.',
                                    'MET amplification was seen in 4 of 75 (5%; 95% CI, 1%-13%).',
                                    ''
                                    ]]

print("Shortest sentences")
print("\nCIViC:\n")
[print(x, "\t", pp.transform([x]))
 for x in sorted(civic, key=len)[:20]]
print("\nAbstracts:\n")
[print(x, "\t", pp.transform([x])) for x in sorted(abstracts, key=len)[:20]]
print("\nPIBOSO other:\n")
[print(x, "\t", pp.transform([x]))
 for x in sorted(piboso_other, key=len)[:20]]
print("\nPIBOSO outcome:\n")
[print(x, "\t", pp.transform([x])) for x in sorted(piboso_outcome, key=len)[:20]]

print("\n----------------------------------------------------------------",
      "\nWord vectors for subsets of CIViC and Abstracts",
      "\n----------------------------------------------------------------\n")

[print(x, "\n", pp.transform([x]), "\n")
 for x in civic_]

[print(x, "\n", pp.transform([x]), "\n") for x in abstracts_]
