from __future__ import absolute_import, division, print_function

import os
import subprocess
import tarfile
import zipfile
import nltk
try:
    import urllib.request as url_lib
except ImportError:
    import urllib as url_lib


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

    # TODO GENIA tagger wrapper

    print("installing GENIA tagger...")

    if not os.path.exists("./semisuper/resources"):
        os.makedirs("./semisuper/resources")

    url_lib.urlretrieve("http://www.nactem.ac.uk/tsujii/GENIA/tagger/geniatagger-3.0.2.tar.gz",
                               "./semisuper/resources/geniatagger-3.0.2.tar.gz")
    tar_ref = tarfile.open("./semisuper/resources/geniatagger-3.0.2.tar.gz")
    tar_ref.extractall("./semisuper/resources/")
    tar_ref.close()
    subprocess.call(["make", "-C", "./semisuper/resources/geniatagger-3.0.2"])

    print("Done installing GENIA tagger.")


def load_HoC():
    """Download and extract Hallmarks of Cancer corpus

    expects cwd to be semisuper root"""

    print("Downloading Hallmarks of Cancer corpus...")

    if not os.path.exists("./semisuper/resources/"):
        os.makedirs("./semisuper/resources")

    url_lib.urlretrieve("https://www.cl.cam.ac.uk/~sb895/HoCCorpus.zip",
                               "./semisuper/resources/corpora/HoCCorpus.zip")
    zip_ref = zipfile.ZipFile("./semisuper/resources/corpora/HoCCorpus.zip", 'r')
    zip_ref.extractall("./semisuper/resources/corpora")
    zip_ref.close()

    print("Done loading HoC.")
    return


if __name__ == "__main__":
    download_nltk_deps()
    load_HoC()
    install_geniatagger()
