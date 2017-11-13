from semisuper import loaders
import nalaf
import nala
import os
import subprocess
import pprint
import multiprocessing as multi

civic, abstracts = loaders.sentences_civic_abstracts()


# ----------------------------------------------------------------

def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)


# ----------------------------------------------------------------
# CCG NLPY


# import ccg_nlpy
# from ccg_nlpy import TextAnnotation, TestPipelineNLPy, TextAnnotation_pb2
# from ccg_nlpy import remote_pipeline

# pipeline = remote_pipeline.RemotePipeline()
# doc = pipeline.doc(civ[0])
# print(doc.get_lemma)
# print(doc.get_pos)
# print(doc.get_ner_ontonotes)
# print(doc.get_stanford_dependency_parse)
# print()


# ----------------------------------------------------------------
# META MAP LITE

def metamaplite(text):

    mmlitepath = file_path("../resources/programs/public_mm_lite/metamaplite.sh")
    pipe_flag = "--"
    sem_types = ["aapp",  # |T116|Amino Acid, Peptide, or Protein
                 "acab",  # |T020|Acquired Abnormality
                 "amas",  # |T087|Amino Acid Sequence
                 "anab",  # |T190|Anatomical Abnormality
                 "anst",  # |T017|Anatomical Structure
                 "antb",  # |T195|Antibiotic
                 "arch",  # |T194|Archaeon
                 "bacs",  # |T123|Biologically Active Substance
                 "bact",  # |T007|Bacterium
                 "bdsu",  # |T031|Body Substance
                 "bdsy",  # |T022|Body System
                 "biof",  # |T038|Biologic Function
                 "blor",  # |T029|Body Location or Region
                 "bmod",  # |T091|Biomedical Occupation or Discipline
                 "bodm",  # |T122|Biomedical or Dental Material
                 "bpoc",  # |T023|Body Part, Organ, or Organ Component
                 "bsoj",  # |T030|Body Space or Junction
                 "carb",  # |T118|Carbohydrate
                 "celc",  # |T026|Cell Component
                 "celf",  # |T043|Cell Function
                 "cell",  # |T025|Cell
                 "cgab",  # |T019|Congenital Abnormality
                 "chem",  # |T103|Chemical
                 "chvf",  # |T120|Chemical Viewed Functionally
                 "chvs",  # |T104|Chemical Viewed Structurally
                 "clas",  # |T185|Classification
                 "clna",  # |T201|Clinical Attribute
                 "clnd",  # |T200|Clinical Drug
                 "comd",  # |T049|Cell or Molecular Dysfunction
                 "crbs",  # |T088|Carbohydrate Sequence
                 "diap",  # |T060|Diagnostic Procedure
                 "drdd",  # |T203|Drug Delivery Device
                 "dsyn",  # |T047|Disease or Syndrome
                 "eico",  # |T111|Eicosanoid
                 "elii",  # |T196|Element, Ion, or Isotope
                 "emst",  # |T018|Embryonic Structure
                 "enzy",  # |T126|Enzyme
                 "euka",  # |T204|Eukaryote
                 "ffas",  # |T021|Fully Formed Anatomical Structure
                 "fndg",  # |T033|Finding
                 "fngs",  # |T004|Fungus
                 "genf",  # |T045|Genetic Function
                 "geoa",  # |T083|Geographic Area
                 "gngm",  # |T028|Gene or Genome
                 "hcro",  # |T093|Health Care Related Organization
                 "hlca",  # |T058|Health Care Activity
                 "hops",  # |T131|Hazardous or Poisonous Substance
                 "horm",  # |T125|Hormone
                 "imft",  # |T129|Immunologic Factor
                 "inch",  # |T197|Inorganic Chemical
                 "inpo",  # |T037|Injury or Poisoning
                 "irda",  # |T130|Indicator, Reagent, or Diagnostic Aid
                 "lbpr",  # |T059|Laboratory Procedure
                 "lbtr",  # |T034|Laboratory or Test Result
                 "lipd",  # |T119|Lipid
                 "mbrt",  # |T063|Molecular Biology Research Technique
                 "medd",  # |T074|Medical Device
                 "mobd",  # |T048|Mental or Behavioral Dysfunction
                 "moft",  # |T044|Molecular Function
                 "mosq",  # |T085|Molecular Sequence
                 "neop",  # |T191|Neoplastic Process
                 "nnon",  # |T114|Nucleic Acid, Nucleoside, or Nucleotide
                 "nsba",  # |T124|Neuroreactive Substance or Biogenic Amine
                 "nusq",  # |T086|Nucleotide Sequence
                 "opco",  # |T115|Organophosphorus Compound
                 "orch",  # |T109|Organic Chemical
                 "orga",  # |T032|Organism Attribute
                 "orgf",  # |T040|Organism Function
                 "orgm",  # |T001|Organism
                 "orgt",  # |T092|Organization
                 "ortf",  # |T042|Organ or Tissue Function
                 "patf",  # |T046|Pathologic Function
                 "phsf",  # |T039|Physiologic Function
                 "phsu",  # |T121|Pharmacologic Substance
                 "podg",  # |T101|Patient or Disabled Group
                 "rcpt",  # |T192|Receptor
                 "rept",  # |T014|Reptile
                 "sosy",  # |T184|Sign or Symptom
                 "strd",  # |T110|Steroid
                 "tisu",  # |T024|Tissue
                 "topp",  # |T061|Therapeutic or Preventive Procedure
                 "virs",  # |T005|Virus
                 "vita",  # |T127|Vitamin
                 ]
    sem_type_flag = "--restrict_to_sts=" + ",".join(sem_types)
    bg_flag = "&"

    p = subprocess.Popen([mmlitepath, pipe_flag, sem_type_flag],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         universal_newlines=True)
    out, err = p.communicate(text)
    p.wait()

    if err:
        return None
    else:
        print("yay")
        return text, out


with multi.Pool(multi.cpu_count() * 4) as p:
    mmapped = p.imap_unordered(metamaplite, civic[:120])

for m in mmapped:
    print(m[0], "\n", m[1], "\n")

print(mmapped)


# ----------------------------------------------------------------
# TAGTOG API

import requests


def tagtog_req(text):
    url = 'https://www.tagtog.net/api/0.1/documents'
    auth = requests.auth.HTTPBasicAuth(username='nachne', password='1quiaerx')
    params = {'project': 'semisuper', 'output': 'ann.json'}
    # text = 'Antibody-dependent cellular cytotoxicity (ADCC), a key effector function for the clinical effectiveness of monoclonal antibodies'
    payload = {'text': text}
    response = requests.put(url, params=params, auth=auth, data=payload)
    print("text:", response, response.text)
    return


for c in civic[1000:1010]:
    tagtog_req(c)
