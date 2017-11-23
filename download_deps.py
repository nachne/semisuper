import nltk

nltk_downloads = ["wordnet",
                  "treebank",
                  "punkt",
                  "dependency_treebank"
                  ]

for d in nltk_downloads:
    nltk.downloads(d)