import pandas as pd
import pickle
from Bio import Medline, Entrez
import multiprocessing as multi
from semisuper import transformers
from semisuper.helpers import flatten
import re
import os.path
import glob

# TODO delete personal e-mail
# needed for querying PubMed API
Entrez.email = 'wackerbm@informatik.hu-berlin.de'


# ----------------------------------------------------------------
# top-level
# ----------------------------------------------------------------

def sentences_civic_abstracts():
    """load CIViC summaries and corresponding PubMed abstracts, return one list of sentences each"""

    civic, abstracts = load_civic_abstracts()

    print("No. of PubMed IDs:\t", len(civic["evidence_statement"]))
    print("No. of abstracts:\t", len(abstracts))

    with multi.Pool(processes=multi.cpu_count()) as p:
        summary_sentences = flatten(p.map(transformers.sentence_tokenize, civic["evidence_statement"]))
        summary_authors2we = [authors2we(s) for s in set(summary_sentences)]
        abstract_sentences = [s for s in flatten(p.map(transformers.sentence_tokenize, abstracts["abstract"]))]

    return summary_authors2we, abstract_sentences


def sentences_piboso_other():
    """load negative sentences (labelled as OTHER) from PIBOSO"""
    return list(sentences_piboso(include=["other"]))


def sentences_piboso_pop_bg_oth():
    """load negative sentences from PIBOSO

    (labelled as OTHER, BACKGROUND, or POPULATION, but not OUTCOME)"""
    return list(sentences_piboso(include=["other", "background", "population"],
                                 exclude=["outcome"]))


def sentences_piboso_outcome():
    """load OUTCOME sentences from PIBOSO"""
    return list(sentences_piboso(include=["outcome"]))


# ----------------------------------------------------------------
# CIViC and PubMed helpers
# ----------------------------------------------------------------

def load_civic_abstracts():
    """load CIViC clinical evidence summaries and corresponding PubMed abstracts

    if already pickled, load from disk, otherwise download"""

    try:
        with open(file_path("./pickles/civic_abstracts.pickle"), "rb") as f:
            (civic, abstracts) = pickle.load(f)
            print("Loaded summaries and abstracts from disk.")

    except IOError:
        print("Downloading summaries...")
        civic = read_civic()

        print("Downloading abstracts...")
        pm_ids = get_pm_ids(civic)
        abstracts = get_abstracts(pm_ids)

        with open(file_path("./pickles/civic_abstracts.pickle"), "wb") as f:
            pickle.dump((civic, abstracts), f)
            print("Download complete, saving to disk.")

    return civic, abstracts


def read_civic(path=""):
    """read CIViC clinical evidence from optional path or download nightly build"""
    if not path:
        path = ("https://civic.genome.wustl.edu/downloads/"
                "nightly/nightly-ClinicalEvidenceSummaries.tsv")
    return pd.read_csv(path, sep='\t')


def get_pm_ids(df):
    """get PubMed IDs in CIViC dataframe"""
    return list({str(idx) for idx in df["pubmed_id"]})


def get_abstracts(idlist):
    """download abstracts from PubMed for a list of PubMed IDs"""
    handle = Entrez.efetch(db="pubmed", id=idlist, rettype="medline", retmode="text")
    records = Medline.parse(handle)
    df = pd.DataFrame(columns=["abstract"])
    for rec in records:
        try:
            df = df.append(pd.DataFrame([[rec["AB"]]], columns=['abstract']), ignore_index=True)
        except Exception:
            pass
    return df


def authors2we(sentence):
    """replaces \"(the) authors\" by \"we\" in a text to mitigate 3rd person perspective of summaries"""
    temp = re.sub("The\s+[Aa]uthors'", "Our", sentence)
    temp = re.sub("the\s+[Aa]uthors'", "our", temp)
    temp = re.sub("by\s+the\s[Aa]uthors", "by us", temp)
    temp = re.sub("The\s+[Aa]uthors", "We", temp)
    return re.sub("(the\s+)?[Aa]uthors", "we", temp)


# ----------------------------------------------------------------
# HoC helpers
# ----------------------------------------------------------------

# TODO decide whether / how to use HoC for training/evaluation
def sentences_HoC():
    """Positive (any HoC category) and negative (uncategorized) sentences from Hallmarks of Cancer corpus"""

    positive = []
    negative = []

    label_re = re.compile("\t\[.*\]")

    for filename in glob.glob(file_path(file_path("./resources/corpora/HoCCorpus/*.txt"))):
        with open(filename, 'r') as f:

            lines = f.read().split('\n')

            for line in lines[1:]:
                if not len(line) or line[0] == '#' or not label_re.findall(line):
                    continue

                label = label_re.findall(line)[0]
                # print("label:", label)

                sentence = label_re.sub("", line)
                # print("sentence:", sentence)

                if label == '\t[]':
                    negative.append(sentence)
                else:
                    positive.append(sentence)

    return positive, negative


# ----------------------------------------------------------------
# PIBOSO helpers
# ----------------------------------------------------------------

def sentences_piboso(include, exclude=None):
    """loads PIBOSO sentences from any included but no excluded class from csv, with normalized abbreviations"""
    piboso = pd.read_csv(file_path("resources/piboso_train.csv"))
    predictions = piboso['Prediction'].values
    texts = piboso['Text'].values

    include_ids = [piboso_category_offset(c) for c in include]
    exclude_ids = [piboso_category_offset(c) for c in exclude] if exclude else []

    for i in range(0, len(predictions) - 6, 6):
        if (any(predictions[i + idx] for idx in include_ids)
            and not
            any(predictions[i + idx] for idx in exclude_ids)):
            yield (transformers.prenormalize(texts[i]))


def piboso_category_offset(category):
    """maps category names to their line offset in PIBOSO file"""
    indices = {"background": 0,
               "intervention": 1,
               "population": 2,
               "outcome": 3,
               "other": 4,
               "study design": 5,
               "study_design": 5}
    return indices[category]


# ----------------------------------------------------------------
# general helpers
# ----------------------------------------------------------------


def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)
