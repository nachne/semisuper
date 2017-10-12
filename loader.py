import pandas as pd
import pickle
from Bio import Medline, Entrez
import multiprocessing as multi
from helpers import flatten
import nltk

# TODO delete personal e-mail
# needed for querying PubMed API
Entrez.email = 'wackerbm@informatik.hu-berlin.de'


# reading CIViC summaries and corresponding abstracts from PubMed

def sentences_civic_abstracts():
    """load CIViC summaries and corresponding abstracts, return one list of sentences each"""

    civic, abstracts = load_civic_abstracts()

    print("No. of PubMed IDs:\t", len(civic["evidence_statement"]))
    print("No. of abstracts:\t", len(abstracts))

    with multi.Pool(processes=multi.cpu_count()) as p:
        summary_sentences = flatten(p.map(nltk.sent_tokenize, civic["evidence_statement"]))
        abstract_sentences = flatten(p.map(nltk.sent_tokenize, abstracts["abstract"]))

    return summary_sentences, abstract_sentences


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
# ----------------------------------------------------------------

def load_civic_abstracts():
    """load CIViC clinical evidence summaries and corresponding PubMed abstracts

    if already pickled, load from disk, otherwise download"""

    try:
        with open("civic_abstracts.pickle", "rb") as f:
            (civic, abstracts) = pickle.load(f)
            print("loaded summaries and abstracts from disk.")

    except:
        print("downloading summaries...")
        civic = read_civic()

        print("downloading abstracts...")
        pm_ids = get_pm_ids(civic)
        abstracts = get_abstracts(pm_ids)

        with open("civic_abstracts.pickle", "wb") as f:
            pickle.dump((civic, abstracts), f)
            print("download complete, saving to disk.")

    return civic, abstracts


def read_civic(path=""):
    """read CIViC clinical evidence from optional path or download nightly build"""
    if not path:
        path = ("https://civic.genome.wustl.edu/downloads/"
                "nightly/nightly-ClinicalEvidenceSummaries.tsv")
    return pd.read_csv(path, sep='\t')


def get_pm_ids(df):
    """get PubMed IDs in CIViC dataframe"""
    return list({str(id) for id in df["pubmed_id"]})


def get_abstracts(idlist):
    """download abstracts from PubMed for a list of PubMed IDs"""
    handle = Entrez.efetch(db="pubmed", id=idlist, rettype="medline", retmode="text")
    records = Medline.parse(handle)
    df = pd.DataFrame(columns=["abstract"])
    for rec in records:
        try:
            df = df.append(pd.DataFrame([[rec["AB"]]], columns=['abstract']), ignore_index=True)
        except:
            pass
    return df


# ----------------------------------------------------------------
# ----------------------------------------------------------------

def sentences_piboso(include=["study design"], exclude=[]):
    """loads PIBOSO sentences where predictions are true for any included class and false for all excluded, from csv"""
    piboso = pd.read_csv("piboso_train.csv")
    predictions = piboso['Prediction'].values
    texts = piboso['Text'].values

    include_ids = [piboso_category_offset(c) for c in include]
    exclude_ids = [piboso_category_offset(c) for c in exclude]

    for i in range(0, len(predictions)-6, 6):
        if (any(predictions[i+id] for id in include_ids)
            and not
            any(predictions[i+id] for id in exclude_ids)):
            yield (texts[i])


def piboso_category_offset(category):
    indices = {"background": 0,
               "intervention": 1,
               "population": 2,
               "outcome": 3,
               "other": 4,
               "study design": 5,
               "study_design": 5}
    return indices[category]
