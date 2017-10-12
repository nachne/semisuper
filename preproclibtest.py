import loader

import ccg_nlpy
from ccg_nlpy import TextAnnotation, TestPipelineNLPy, TextAnnotation_pb2
from ccg_nlpy import remote_pipeline



civ, abs = loader.sentences_civic_abstracts()

# pipeline = remote_pipeline.RemotePipeline()
# doc = pipeline.doc(civ[0])
# print(doc.get_lemma)
# print(doc.get_pos)
# print(doc.get_ner_ontonotes)
# print(doc.get_stanford_dependency_parse)
# print()

