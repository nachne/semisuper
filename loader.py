import pandas as pd
import pickle
from Bio import Medline, Entrez

# TODO delete personal e-mail
# needed for querying PubMed API
Entrez.email = 'wackerbm@informatik.hu-berlin.de'

# reading CIViC summaries and corresponding abstracts from PubMed

def load_civic_abstracts():
    try:
        with open("civic_abstracts.pickle", "rb") as f:
            (civic, abstracts) = pickle.load(f)
            print("loaded summaries and abstracts from disk.")
    except:
        print("downloading summaries...")
        civic = read_civic()
        print("downloading abstracts...")
        abstracts = get_abstracts(get_pm_ids(civic))
        print("download complete.")
        with open("civic_abstracts.pickle", "wb") as f:
            pickle.dump((civic, abstracts), f)
    return civic, abstracts


def read_civic(path=""):
    if not path:
        path = ("https://civic.genome.wustl.edu/downloads/"
                "nightly/nightly-ClinicalEvidenceSummaries.tsv")
    return pd.read_csv(path, sep='\t')

def get_pm_ids(df):
    return list(map(str, df["pubmed_id"]))

def get_abstracts(idlist):
    handle = Entrez.efetch(db="pubmed", id=idlist, rettype="medline", retmode="text")
    records = Medline.parse(handle)
    df = pd.DataFrame(columns=["abstract"])
    for rec in records:
        try:
            df = df.append(pd.DataFrame([[rec["AB"]]], columns=['abstract']), ignore_index=True)
        except:
            pass
    return df