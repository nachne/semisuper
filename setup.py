from __future__ import absolute_import, division, print_function

# from setuptools import setup

import os
import sys
import pip
import subprocess
import tarfile
import zipfile
import nltk

try:
    import urllib.request as url_lib
except ImportError:
    import urllib as url_lib


def pipinstall(pkg_name):
    """install package with pip subprocess"""
    pip.main(["install", "--user", pkg_name])


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


def install_geniatagger():
    """Download and make geniatagger-3.0.2

    expects cwd to be semisuper root"""

    if os.path.isfile("./semisuper/resources/geniatagger-3.0.2/geniatagger"):
        return

    mkdir("./semisuper/resources")

    print("installing GENIA tagger...")

    url_lib.urlretrieve("http://www.nactem.ac.uk/tsujii/GENIA/tagger/geniatagger-3.0.2.tar.gz",
                        "./semisuper/resources/geniatagger-3.0.2.tar.gz")
    tar_ref = tarfile.open("./semisuper/resources/geniatagger-3.0.2.tar.gz")
    tar_ref.extractall("./semisuper/resources/")
    tar_ref.close()
    subprocess.call(["make", "-C", "./semisuper/resources/geniatagger-3.0.2"])

    pipinstall("https://github.com/d2207197/geniatagger-python/archive/master.zip")

    print("Done installing GENIA tagger.")


def load_HoC():
    """Download and extract Hallmarks of Cancer corpus

    expects cwd to be semisuper root"""

    if os.path.isfile("./semisuper/resources/HoCCorpus/1280402.txt"):
        return

    print("Downloading Hallmarks of Cancer corpus...")

    mkdir("./semisuper/resources")

    url_lib.urlretrieve("https://www.cl.cam.ac.uk/~sb895/HoCCorpus.zip",
                        "./semisuper/resources/corpora/HoCCorpus.zip")
    zip_ref = zipfile.ZipFile("./semisuper/resources/corpora/HoCCorpus.zip", 'r')
    zip_ref.extractall("./semisuper/resources/corpora")
    zip_ref.close()

    print("Done loading HoC.")
    return


def install_all():
    for path in ["./semisuper/resources", "./semisuper/silver_standard", "./semisuper/pickles"]:
        mkdir(path)

    for pkg in ["biopython==1.70", "nltk==3.2.4", "numpy==1.13.3", "pandas==0.20.3", "scikit-learn==0.19.1",
                "scipy==0.19.1", "Unidecode==0.4.21"]:
        pipinstall(pkg)

    download_nltk_deps()
    load_HoC()
    install_geniatagger()


if __name__ == "__main__":
    # setup(install_requires=['scikit-learn', 'scipy', 'numpy', 'pandas', 'nltk',
    #                         'geniatagger-python', 'unidecode', 'gensim'])

    install_all()
