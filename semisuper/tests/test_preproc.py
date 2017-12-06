import random

from semisuper import loaders, transformers
from semisuper.helpers import identity, densify
from basic_pipeline import identitySelector
from semisuper.basic_pipeline import factorization, vectorizer, percentile_selector
from semisuper.transformers import TokenizePreprocessor, TextStats, FeatureNamePipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import time
import sys

# ----------------------------------------------------------------
# Data
# ----------------------------------------------------------------

civic, abstracts = loaders.sentences_civic_abstracts()
hocpos, hocneg = loaders.sentences_HoC()
piboso_other = loaders.sentences_piboso_other()
piboso_outcome = loaders.sentences_piboso_outcome()


civic_, _ = train_test_split(civic, test_size=0.5)
abstracts_, _ = train_test_split(abstracts, test_size=0.5)
hocpos_, _ = train_test_split(hocpos, test_size=0.5)
hocneg_, _ = train_test_split(hocneg, test_size=0.5)

# ----------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------

min_df_word = 20
min_df_char = 50
n_components = 100
print("min_df: \tword:", min_df_word, "\tchar:", min_df_char, "\tn_components:", n_components)

v = vectorizer(chargrams=(2, 6), min_df_char=min_df_char, wordgrams=(1, 4), min_df_word=min_df_word, lemmatize=True,
               rules=True, max_df=0.95)
s = factorization('TruncatedSVD')

pppl = Pipeline([
    ('vectorizer', v),
    ('selector', s)
])

# ----------------------------------------------------------------
# Regex tests
# ----------------------------------------------------------------



pp = TokenizePreprocessor()

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

print(transformers.sentence_tokenize(
        "He is going to be there, i.e. Dougie is going to be there.I can not "
        "wait to see how he's RS.123 doing. Conf. fig. 13 for additional "
        "information. E.g."
        "if you want to know the time, you should take b. "
        "Beispieltext ist schön. first. 10. we are going to cowabunga. let there be rainbows."))

[print(x, pp.transform([x])) for x in ['Janus kinase 2 JAK2 tyrosine kinase',
                                       'CLINICALTRIALSGOV: NCT00818441.',
                                       'clinicaltrials.gov',
                                       'genetic/mrna',
                                       'C-->A G-->T',
                                       'don\'t go wasting your emotions',
                                       'Eight inpatient rehabilitation facilities.',
                                       'There were 67 burst fractures, 48 compression fractures and 21 fracture dislocations, 8 flexion distraction fractures and 6 flexion rotation injuries.',
                                       'MET amplification was seen in 4 of 75 (5%; 95% CI, 1%-13%).',
                                       ''
                                       ]]

print("\nShortest sentences")
print("\nCIViC:\n")
[print(x, "\t", pp.transform([x])) for x in sorted(civic, key=len)[:20]]
print("\nAbstracts:\n")
[print(x, "\t", pp.transform([x])) for x in sorted(abstracts, key=len)[:20]]
print("\nHoC pos:\n")
[print(x, "\t", pp.transform([x])) for x in sorted(hocpos, key=len)[:20]]
print("\nHoC neg:\n")
[print(x, "\t", pp.transform([x])) for x in sorted(hocneg, key=len)[:20]]
print("\nPIBOSO outcome:\n")
[print(x, "\t", pp.transform([x])) for x in sorted(piboso_outcome, key=len)[:20]]
print("\nPIBOSO other:\n")
[print(x, "\t", pp.transform([x])) for x in sorted(piboso_other, key=len)[:20]]

print("\nLongest sentences")
print("\nCIViC:\n")
[print(x, "\t", pp.transform([x])) for x in sorted(civic, key=len)[-20:]]
print("\nAbstracts:\n")
[print(x, "\t", pp.transform([x])) for x in sorted(abstracts, key=len)[-20:]]
print("\nHoC pos:\n")
[print(x, "\t", pp.transform([x])) for x in sorted(hocpos, key=len)[-20:]]
print("\nHoC neg:\n")
[print(x, "\t", pp.transform([x])) for x in sorted(hocneg, key=len)[-20:]]
print("\nPIBOSO outcome:\n")
[print(x, "\t", pp.transform([x])) for x in sorted(piboso_outcome, key=len)[-20:]]
print("\nPIBOSO other:\n")
[print(x, "\t", pp.transform([x])) for x in sorted(piboso_other, key=len)[-20:]]

print("\n----------------------------------------------------------------",
      "\nWord vectors for subsets of CIViC and Abstracts",
      "\n----------------------------------------------------------------\n")

[print(x, "\n", pp.transform([x]), "\n")
 for x in civic_[1:10]]

[print(x, "\n", pp.transform([x]), "\n") for x in abstracts_[1:10]]

# ----------------------------------------------------------------
# vectorization and selection performance
# ----------------------------------------------------------------

civic_, _ = train_test_split(civic, test_size=0.75)
abstracts_, _ = train_test_split(abstracts, test_size=0.75)
hocpos_, _ = train_test_split(hocpos, test_size=0.75)
hocneg_, _ = train_test_split(hocneg, test_size=0.75)

# test_corpus = np.concatenate((civic_, abstracts_, hocpos_, hocneg_))
test_corpus = np.concatenate((civic, abstracts, hocpos, hocneg))

print("Training preprocessing pipeline")

start_time = time.time()
v.fit(test_corpus)
print("vectorizing took", time.time() - start_time, "secs.")

start_time = time.time()
vectorized_corpus = v.transform(test_corpus)
print("transforming took", time.time() - start_time, "secs. \t", np.shape(vectorized_corpus)[1], "features")

min_df_word = 0.001
min_df_char = 0.001
n_components = [10, 100, 1000]
print("min_df: \tword:", min_df_word, "\tchar:", min_df_char, "\tn_components:", n_components)

v = vectorizer(chargrams=(2, 6), min_df_char=min_df_char, wordgrams=(1, 4), min_df_word=min_df_word, lemmatize=True,
               rules=True, max_df=0.95)

# TruncatedSVD:                 ok              3.5min at cutoff 50/100
# LatentDirichletAllocation:    few topics!
# 'NMF':                        slow
sparse = []  # ['TruncatedSVD']
# 'PCA':                        RAM
# SparsePCA:
# IncrementalPCA:
# Factor Analysis:              slow
dense = ['PCA']

# NOT:
# MiniBatchSparsePCA:           >15h/CPU, 16GB

# start_time = time.time()
# LatentDirichletAllocation(n_topics=10, n_jobs=-1).fit(vectorized_corpus)
# print("fitting LDA took", time.time() - start_time, "secs")


for sel in sparse:
    for n_comps in n_components:
        start_time = time.time()
        factorization(sel, n_comps).fit(vectorized_corpus)
        print("fitting", sel, "with", n_comps, "components took", time.time() - start_time, "secs")

for sel in dense:
    for n_comps in n_components:
        start_time = time.time()
        factorization(sel, n_comps).fit(densify(vectorized_corpus))
        print("fitting", sel, "with", n_comps, "components ok", time.time() - start_time, "secs")
