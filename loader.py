import pandas as pd
import pickle
from Bio import Medline, Entrez

# TODO delete personal e-mail
# needed for querying PubMed API
Entrez.email = 'wackerbm@informatik.hu-berlin.de'

# reading CIViC summaries and corresponding abstracts from PubMed

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