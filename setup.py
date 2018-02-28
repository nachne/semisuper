# coding=utf-8
from __future__ import absolute_import, division, print_function

# from setuptools import setup

import os
import pip
import zipfile
import nltk

try:
    import urllib.request as url_lib
except ImportError:
    import urllib as url_lib


def pipinstall(pkg_name):
    """install package with pip subprocess"""
    pip.main(["install", pkg_name])


def mkdir(path):
    """make directory if not present"""
    if not os.path.exists(path):
        os.makedirs(path)


def download_nltk_deps():
    nltk_downloads = ["wordnet",
                      "treebank",
                      "punkt",
                      "dependency_treebank",
                      "stopwords",
                      "averaged_perceptron_tagger"
                      ]

    for d in nltk_downloads:
        nltk.download(d)


def install_all():
    for path in ["./semisuper/silver_standard", "./semisuper/pickles"]:
        mkdir(path)

    for pkg in ["biopython==1.70", "numpy==1.13.3", "pandas==0.20.3", "scikit-learn==0.19.1",
                "scipy==0.19.1", "Unidecode==0.4.21"]:
        pipinstall(pkg)

    download_nltk_deps()


if __name__ == "__main__":
    install_all()
