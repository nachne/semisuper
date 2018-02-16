from __future__ import absolute_import, division, print_function

import pickle
import os
import time
import multiprocessing
import random
from sys import argv, exit

from semisuper import loaders, helpers
import key_sentence_predictor


def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)


def main(max_abstracts=400, n_jobs=12, batch_size=400):
    """loads max_abstracts abstracts for processing, classifies sentences and returns html text"""

    try:
        with open(file_path("semisuper/pickles/sent_test_abstract_dicts.pickle"), "rb") as f:
            abstracts = pickle.load(f)
            abstracts = random.sample(abstracts, min(len(abstracts), max_abstracts))
    except:
        abstracts = loaders.fetch(loaders.get_pmids_from_query(term="cancer", max_ids=max_abstracts))
        abstracts = [{"pmid": a["PMID"], "abstract": a["AB"], "title": a["TI"], "authors": a["AU"]}
                     for a in random.sample(abstracts, min(len(abstracts), max_abstracts))
                     if all([a.get("AB"), a.get("PMID"), a.get("AU"), a.get("TI")])]
        with open(file_path("semisuper/pickles/sent_test_abstract_dicts.pickle"), "wb") as f:
            pickle.dump(abstracts, f)

    predictor = key_sentence_predictor.KeySentencePredictor(batch_size=batch_size)

    if max_abstracts < 2 * batch_size:
        results = predictor.transform(abstracts)
    else:
        n_jobs = min(n_jobs, multiprocessing.cpu_count())
        with multiprocessing.Pool(n_jobs) as p:
            start_time = time.time()
            results = helpers.merge_dicts(p.map(predictor.transform,
                                                helpers.partition(abstracts, len(abstracts) // n_jobs),
                                                chunksize=1))

            print("Preprocessing and predicting relevant sentences for", len(abstracts), " abstracts",
                  "took", time.time() - start_time, "seconds")

    html = make_html(abstracts, results)

    with open(file_path("./demo.html"), "w") as f:
        f.write(html)
    print("Wrote results to ./demo.html")

    return html


def make_html(abstracts, relevant):
    """returns html file as a string where all articles have their key sentences highlighted"""

    articles = []
    for a in abstracts:
        articles.append(make_article(a, relevant[a["pmid"]]))

    html = "<!DOCTYPE html>\n<meta charset=\"UTF-8\">\n<html>\n<body style=\"font-family: sans-serif\">" \
           "\n<h1>Demo</h1>\n\n" \
           "{}" \
           "\n\n</body></html>".format("\n\n".join(articles))

    return html


def make_article(abstract, hits):
    """returns html <article> string for abstract where key sentences in hits are surrounded by spans marking
    confidence"""

    text = abstract["abstract"]
    for (start, end, score) in sorted(hits, reverse=True):
        # starting from the back to preserve indices, surround positions by spans with score as bg yellow's opacity
        text = "{}\n<span style=\"background-color: rgba(255,255,0,{})\">{}</span>\n{}".format(
                text[:start],
                (0.3 + min(score, 0.5) / 0.7),
                text[start:end],
                text[end:]
        )

    article = "<article>\n<h2>{}</h2>\n<h3>{}<br>PMID: {}</h3>\n<p>\n{}\n</p>\n</article>".format(
            abstract["title"],
            ", ".join(abstract["authors"]),
            abstract["pmid"],
            text
    )

    return article


if __name__ == "__main__":

    if len(argv) == 2:
        html = main(max_abstracts=int(argv[1]))
    elif len(argv) == 3:
        html = main(max_abstracts=int(argv[1]), n_jobs=int(argv[2]))
    elif len(argv) == 4:
        html = main(max_abstracts=int(argv[1]), n_jobs=int(argv[2]), batch_size=int(argv[3]))
    else:
        html = main()

    # print(html)

    exit(0)
