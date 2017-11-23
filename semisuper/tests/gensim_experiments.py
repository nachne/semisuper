import string
from collections import Counter
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import gensim
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from semisuper import loaders, transformers

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

# make smaller
# civic = random.sample(civic, 100)
# abstracts = random.sample(abstracts, 100)

corpus = (
    []
    + civic
    + abstracts
    + hocpos
    + hocneg
    + piboso_outcome
    + piboso_other
)

pp = transformers.TokenizePreprocessor(rules=False, lemmatize=False)
corp_tokenized = pp.transform(corpus)

# --------------------------------
# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(corp_tokenized)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in corp_tokenized]

# Creating the object for LDA model using gensim library
# Running and Trainign LDA model on the document term matrix.
lda = LdaMulticore(doc_term_matrix, num_topics=12, id2word=dictionary, passes=50)

[print(x) for x in lda.print_topics(num_topics=-1, num_words=20)]


# --------------------------------

def topics2label(topics):
    return max(topics, key=itemgetter(1))[0]

def label_dist(corpus, dict, ldamodel):
    labels = [topics2label(ldamodel[x]) for x in
              [dict.doc2bow(doc) for doc in corpus]]

    freqs = Counter(labels)

    return [(lbl, freqs[lbl], freqs[lbl] / len(corpus)) for lbl in sorted(freqs)]


print("\nCIViC:")
[print(x, end="\t") for x in (label_dist(pp.transform(civic), dictionary, lda))]

print("\nAbstracts:")
[print(x, end="\t") for x in (label_dist(pp.transform(abstracts), dictionary, lda))]

print("\nHoC pos:")
[print(x, end="\t") for x in (label_dist(pp.transform(hocpos), dictionary, lda))]

print("\nHoC neg:")
[print(x, end="\t") for x in (label_dist(pp.transform(hocneg), dictionary, lda))]

print("\nPIBOSO outcome:")
[print(x, end="\t") for x in (label_dist(pp.transform(piboso_outcome), dictionary, lda))]

print("\nPIBOSO other:")
[print(x, end="\t") for x in (label_dist(pp.transform(piboso_other), dictionary, lda))]

# --------------------------------
# obsolete tutorial stuff

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
