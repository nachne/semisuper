import nltk

nltk_downloads = ["wordnet",
                  "treebank",
                  "punkt",
                  "dependency_treebank",
                  "stopwords",
                  "averaged_perceptron_tagger"
                  ]

for d in nltk_downloads:
    nltk.download(d)