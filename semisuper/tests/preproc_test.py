from semisuper import loaders
from semisuper.transformers import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
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

print("CIViC sentences:", len(civic))
print("Abstract sentences:", len(abstracts))
print("HoC positive sentences:", len(hocpos))
print("HoC negative sentences:", len(hocneg))
print("PIBOSO outcome sentences:", len(piboso_outcome))
print("PIBOSO other sentences:", len(piboso_other))

# ----------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------

# min_df_word = 20
# min_df_char = 50
# n_components = 100
# print("min_df: \tword:", min_df_word, "\tchar:", min_df_char, "\tn_components:", n_components)
#
# v = vectorizer(chargrams=(2, 6), min_df_char=min_df_char, wordgrams=(1, 4), min_df_word=min_df_word, ner=True,
#                rules=True, max_df=0.95)
# s = factorization('TruncatedSVD')
#
# pppl = Pipeline([
#     ('vectorizer', v),
#     ('selector', s)
# ])

# ----------------------------------------------------------------
# Regex tests
# ----------------------------------------------------------------


pp = Pipeline([("normalizer", TextNormalizer()),
               ("preprocessor", TokenizePreprocessor(rules=True, genia=True, ner=True, pos=True))])

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
                         "wild-type wild type wildtype",
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

[print(x, pp.transform([x])) for x in ['Janus kinase 2 JAK2 tyrosine kinase',
                                       'Proc. cancer-related specific recurrence-free '
                                       'person-years phase-3 open-label --&gt',
                                       'CLINICALTRIALSGOV: NCT00818441.',
                                       'clinicaltrials.gov',
                                       'genetic/mrna',
                                       'C-->A G-->T',
                                       '(c)2015 AACR',
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

exmpl_abs = "Activating mutations in tyrosine kinases have been identified in hematopoietic and nonhematopoietic malignancies. Recently, we and others identified a single recurrent somatic activating mutation (JAK2V617F) in the Janus kinase 2 (JAK2) tyrosine kinase in the myeloproliferative disorders (MPDs) polycythemia vera, essential thrombocythemia, and myeloid metaplasia with myelofibrosis. We used direct sequence analysis to determine if the JAK2V617F mutation was present in acute myeloid leukemia (AML), chronic myelomonocytic leukemia (CMML)/atypical chronic myelogenous leukemia (aCML), myelodysplastic syndrome (MDS), B-lineage acute lymphoblastic leukemia (ALL), T-cell ALL, and chronic lymphocytic leukemia (CLL). Analysis of 222 patients with AML identified JAK2V617F mutations in 4 patients with AML, 3 of whom had a preceding MPD. JAK2V617F mutations were identified in 9 (7.8%) of 116 CMML/a CML samples, and in 2 (4.2%) of 48 MDS samples. We did not identify the JAK2V617F disease allele in B-lineage ALL (n = 83), T-cell ALL (n = 93), or CLL (n = 45). These data indicate that the JAK2V617F allele is present in acute and chronic myeloid malignancies but not in lymphoid malignancies."

[print(x, pp.transform([x])) for x in sentence_tokenize(exmpl_abs)]

sys.exit(0)

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
dense = []  # ['PCA']

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
