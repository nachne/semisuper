import pandas as pd
import numpy as np
from Bio import Medline, Entrez
import urllib.request

#

Entrez.email = 'wackerbm@informatik.hu-berlin.de'

#

def read_civic(path=""):
    if (path == ""):
        path = ("https://civic.genome.wustl.edu/downloads/"
            "nightly/nightly-ClinicalEvidenceSummaries.tsv")
    names = []
    return pd.read_csv(path, sep='\t')


def get_medline(idlist):
    handle = Entrez.efetch(db="pubmed", id=idlist, rettype="medline", retmode="text")
    records = Medline.parse(handle)
    df = pd.DataFrame(columns=["abstract"])
    for rec in records:
        try:
            df = df.append(pd.DataFrame([[rec["AB"]]], columns=['abstract']), ignore_index=True)
        except:
            pass
    return df

#

civic = read_civic("/Users/emperor/HU/BA/civic/nightly-ClinicalEvidenceSummaries.tsv")
# print(civic.describe())

pm_ids = list(map(str, civic["pubmed_id"]))
# print("No. of PubMed IDs: ", len(pm_ids))

abstracts = get_medline(pm_ids[0:100])
# print("With abstracts: ", len(abstracts))
# print(abstracts["abstract"])




# obsolet

def load_url(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as conn:
        return conn.read()

def get_pubmed(id, timeout=60):
    prefix = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        "?db=pubmed&retmode=text&rettype=medline&id=")
    text = load_url(prefix + id, timeout)
    return medline_to_df(text)

def medline_to_df(text):
    record = Medline.read(text)
    return record