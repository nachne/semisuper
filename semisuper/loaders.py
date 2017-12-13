import glob
import multiprocessing as multi
import os.path
import pickle
import re

import pandas as pd
from Bio import Medline, Entrez

from semisuper import transformers, helpers
from semisuper.helpers import flatten
from datetime import datetime

# needed for querying PubMed API

Entrez.email = 'wackerbm@informatik.hu-berlin.de'
MIN_LEN = 8


# ----------------------------------------------------------------
# top-level
# ----------------------------------------------------------------

def sentences_civic_abstracts(verbose=False):
    """load CIViC summaries and corresponding PubMed abstracts, return one list of sentences each"""

    civic, abstracts = load_civic_abstracts()

    if verbose:
        print("No. of PubMed IDs:\t", len(civic["evidence_statement"]))
        print("No. of abstracts:\t", len(abstracts))

    # TODO: check CPU count
    with multi.Pool(processes=min(multi.cpu_count(), 24)) as p:
        summary_sentences = flatten(p.map(transformers.sentence_tokenize, civic["evidence_statement"]))
        summary_authors2we = [authors2we(s) for s in set(summary_sentences) if len(s) >= MIN_LEN]
        abstract_sentences = [s for s in flatten(p.map(transformers.sentence_tokenize, abstracts["abstract"]))]

    return summary_authors2we, abstract_sentences


def abstract_pmid_pos_sentences(abstracts=None):
    if abstracts is None:
        _, abstracts = load_civic_abstracts()

    with multi.Pool(processes=multi.cpu_count()) as p:
        result = flatten(p.map(pmid_pos_sentences, zip(abstracts["pmid"], abstracts["abstract"])))

    return result


def abstract_pmid_pos_sentences_query(max_ids=10000, term="cancer OR oncology OR mutation", anew=False, path=None):
    if path is None:
        path = file_path("./pickles/pubmed_dump_" + term + ".pickle")

    if not anew:
        try:
            with open(path, "rb") as f:
                dump = pickle.load(f)
                return dump[min(len(dump), max_ids)]
        except(Exception):
            pass

    print("Retrieving PubMed abstracts for query \"{}\" (max. {})".format(term, max_ids))

    idlist = get_pmids_from_query(max_ids=max_ids, term=term)
    abstracts = get_abstracts(idlist)

    print("No. of fetched abstracts:", len(abstracts))

    pmid_pos_sents = abstract_pmid_pos_sentences(abstracts)

    with open(path, "wb") as f:
        pickle.dump(pmid_pos_sents, f)

    return pmid_pos_sents


def abstract_pmid_pos_sentences_idlist(idlist=123):
    abstracts = get_abstracts(idlist)

    return abstract_pmid_pos_sentences(abstracts)


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

def load_civic_abstracts(verbose=False):
    """load CIViC clinical evidence summaries and corresponding PubMed abstracts

    if already pickled, load from disk, otherwise download"""

    try:
        with open(file_path("./pickles/civic_abstracts.pickle"), "rb") as f:
            (civic, abstracts) = pickle.load(f)
            if verbose:
                print("Loaded summaries and abstracts from disk.")

    except IOError:
        print("Downloading summaries...")
        civic = read_civic()

        pmids = get_pmids_from_df(civic)
        print("Downloading abstracts... (", len(pmids), "unique PMIDs )")
        abstracts = get_abstracts(pmids)

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


def get_pmids_from_df(df):
    """get PubMed IDs in CIViC dataframe"""
    return list({str(idx) for idx in df["pubmed_id"]})


def get_pmids_from_query(max_ids=10000, mindate="1900/01/01", term="cancer"):
    retstart = 0
    idlist = []

    last_len = 1
    handle = None

    retmax = min(100000, max_ids)

    while last_len > 0 and len(idlist) < max_ids:
        handle = Entrez.esearch(db="pubmed", term="(" + term + ") AND (hasabstract[text])",
                                retstart=retstart, retmax=retmax, mindate=mindate)
        new_ids = list(Entrez.read(handle)["IdList"])

        idlist += new_ids
        last_len = len(new_ids)
        retstart += last_len

    if handle:
        handle.close()

    print("No. of fetched PMIDs:", len(idlist))
    return idlist


def get_abstracts(idlist):
    """download abstracts from PubMed for a list of PubMed IDs"""

    records = []

    with multi.Pool(multi.cpu_count() * 4) as p:
        records = p.map(efetch, helpers.partition(idlist, 4000))

    df = pd.DataFrame(columns=["pmid", "title", "abstract"])  # , "authors", "date",
    for rec in records:
        try:
            pmid = rec["PMID"]
            ab = rec["AB"]
            title = rec["TI"]
            df = df.append(pd.DataFrame([[pmid, title, ab]],
                                        columns=["pmid", "title", "abstract"]),
                           ignore_index=True)
        except Exception:
            pass

    return df


def efetch(idlist):
    handle = Entrez.efetch(db="pubmed", id=idlist, rettype="medline", retmode="text")
    records = list(Medline.parse(handle))
    handle.close()
    return records


def pmid_pos_sentences(pmid_abstract):
    """return list of tuples (pmid, sentence position, sentence)"""
    pmid = pmid_abstract[0]
    sentences = transformers.sentence_tokenize(pmid_abstract[1])
    count = len(sentences)

    pmid_pos_s = []

    for i in range(count):
        pmid_pos_s.append([pmid, (i / count), sentences[i]])

    return pmid_pos_s


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

                if label != '\t[]' and len(sentence) >= MIN_LEN:
                    positive.append(sentence)
                elif label == '\t[]':
                    negative.append(sentence)

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
    indices = {"background"  : 0,
               "intervention": 1,
               "population"  : 2,
               "outcome"     : 3,
               "other"       : 4,
               "study design": 5,
               "study_design": 5}
    return indices[category]


# ----------------------------------------------------------------
# general helpers
# ----------------------------------------------------------------


def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)
