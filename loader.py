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

    # print(civic.describe())
    # print(civic["evidence_statement"][0:100])

    # pm_ids = get_pm_ids(civic)
    print("No. of PubMed IDs:\t", len(civic["evidence_statement"]))

    # abstracts = get_abstracts(pm_ids[0:100]) #[0:100] #for quick test
    print("No. of abstracts:\t", len(abstracts))
    # print(abstracts["abstract"])

    with multi.Pool(processes=multi.cpu_count()) as p:
        summary_sentences = flatten(p.map(nltk.sent_tokenize, civic["evidence_statement"]))
        abstract_sentences = flatten(p.map(nltk.sent_tokenize, abstracts["abstract"]))

    return summary_sentences, abstract_sentences


def sentences_piboso_other():
    """load negative sentences (labelled only as OTHER in PIBOSO) from pickle or csv"""
    try:
        with open("piboso_other.pickle", "rb") as f:
            piboso_other = pickle.load(f)
            print("loaded negative sentences from disk.")

    except:
        print("loading negative sentences...")

        piboso = pd.read_csv("piboso_train.csv")
        piboso_other = []

        predictions = piboso['Prediction'].values
        texts = piboso['Text'].values

        for i in range(0, len(piboso) - 6, 6):
            # every sentence appears once for each label; only the 4th should be true
            if (list(predictions[i:i + 6]) == [0, 0, 0, 0, 1, 0]):
                piboso_other.append(texts[i])

        with open("piboso_other.pickle", "wb") as f:
            pickle.dump(piboso_other, f)
            print("negative sentences complete, saving to disk.")

    return piboso_other


def sentences_piboso_pop_bg_oth():
    """load negative sentences in PIBOSO from pickle or csv

    (labelled as OTHER, BACKGROUND, or POPULATION, but not OUTCOME)"""
    try:
        with open("piboso_bg_pop_oth.pickle", "rb") as f:
            piboso_neg = pickle.load(f)
            print("loaded negative sentences from disk.")

    except:
        print("loading negative sentences...")

        piboso = pd.read_csv("piboso_train.csv")
        piboso_neg = []

        predictions = piboso['Prediction'].values
        texts = piboso['Text'].values

        for i in range(0, len(predictions) - 6, 6):
            # every sentence appears once for each label;
            # 3rd (outcome) should be false,
            # background, population or other should be true
            if (predictions[3] == 0 and
                    (predictions[0] == 1 or
                             predictions[2] == 1 or
                             predictions[4] == 1)):
                piboso_neg.append(texts[i])

        with open("piboso_bg_pop_oth.pickle", "wb") as f:
            pickle.dump(piboso_neg, f)
            print("negative sentences complete, saving to disk.")

    return piboso_neg


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
    return [str(id) for id in df["pubmed_id"]]


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
