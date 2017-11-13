from semisuper import loaders
import nalaf
import nala

civic, abstracts = loaders.sentences_civic_abstracts()

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
# NALA


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