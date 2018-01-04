import multiprocessing as multi
import os.path
import pickle
import re
import string
from functools import partial

import pandas as pd
from nltk import TreebankWordTokenizer
from nltk import pos_tag
from nltk import sent_tokenize
from nltk.corpus import stopwords, wordnet

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, PCA, SparsePCA, FactorAnalysis
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectPercentile, f_classif, mutual_info_classif, SelectFromModel
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC
from unidecode import unidecode
from geniatagger import GeniaTagger

from semisuper import helpers

MIN_LEN = 8


def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)


# ----------------------------------------------------------------
# Tokenization
# ----------------------------------------------------------------

# TODO several or only one instance?
tagger = GeniaTagger(file_path("./resources/geniatagger-3.0.2/geniatagger"))


class TokenizePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, punct=None, lower=True, strip=True, ner=False, rules=True):
        self.lower = lower
        self.strip = strip
        self.punct = punct or set(string.punctuation).difference(set('%='))

        self.rules = rules
        self.ner = ner

        self.genia = True

        self.splitters = re.compile("-->|->|[-/.,|<>]")

        self.dict_mapper = HypernymMapper()
        self.tokenizer = TreebankWordTokenizer()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [", ".join(doc) for doc in X]

    def transform(self, X):
        return [self.representation(sentence) for sentence in X]

    def representation(self, sentence):

        tokens_tags = list(self.tokenize(sentence))
        if not tokens_tags:
            return ["_empty_sentence_"]
        return [t for (t, p) in tokens_tags]

    def tokenize(self, sentence):
        """break sentence into pos-tagged tokens; normalize and split on hyphens"""

        # extremely short sentences shall be ignored by next steps
        if len(sentence) < MIN_LEN:
            return []

        if self.genia:
            for token, base, pos, chunk, ne in tagger.parse(sentence):
                # Apply preprocessing to the token
                token_nrm = self.normalize_token(token, pos)

                # TODO split into subtokens or not?

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token_nrm):
                    continue
                yield token_nrm, pos
        else:
            for token, pos in pos_tag(self.tokenizer.tokenize(sentence)):
                # Apply preprocessing to the token
                token_nrm = self.normalize_token(token, pos)
                subtokens = [self.normalize_token(t, pos) for t in self.splitters.split(token_nrm)]

                for subtoken in subtokens:
                    # If punctuation, ignore token and continue
                    if all(char in self.punct for char in token):
                        continue
                    yield subtoken, pos

    def normalize_token(self, token, tag):
        # Apply preprocessing to the token
        token = token.strip() if self.strip else token
        token = token.strip('*') if self.strip else token
        token = token.strip('.') if self.strip else token

        token = token.lower() if self.lower else token

        if self.ner:
            token = self.dict_mapper.replace(token)
        if self.rules:
            token = map_regex_concepts(token)

        return token


def sentence_tokenize(text):
    """tokenize text into sentences after simple normalization"""

    # return PubMedSentenceTokenizer().tokenize(prenormalize(text))

    return sent_tokenize(prenormalize(text))


# ----------------------------------------------------------------
# Normalization, RegEx based replacement


class TextNormalizer(BaseEstimator, TransformerMixin):
    """replaces all non-ASCII characters by approximations, all numbers by 1"""

    def __init__(self, individual_digits=True):
        return

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        return np.array([unidecode(x) for x in X])  # replace all numbers by "1"

        # version without number replacement
        return np.array([unidecode(x) for x in X])


class DigitNormalizer(BaseEstimator, TransformerMixin):
    """replaces all non-ASCII characters by approximations, all numbers by 1"""

    def __init__(self, individual_digits=True):
        if individual_digits:
            self.num = re.compile("\d")
        else:
            self.num = re.compile("(\d+(,\d\d\d)*)|(\d*\.\d+)+")
        return

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        # TODO check if these help
        return np.array([self.num.sub("1", x) for x in X])  # replace all numbers by "1"


def map_regex_concepts(token):
    """replaces abbreviations matching simple REs, e.g. for numbers, percentages, gene names, by class tokens"""

    for regex, repl in regex_concept_dict:
        if regex.findall(token):
            return repl

    return token


regex_concept_dict = [
    # biomedical
    (re.compile("\w+inib$"), "_chemical_"),
    (re.compile("\w+[ui]mab$"), "_chemical_"),
    (re.compile("->|-->"), "_replacement_"),
    # (re.compile("^(PFS|pfs)$"), "progression-free survival"),

    # number-related concepts
    (re.compile("^[Pp]([=<>≤≥]|</?=|>/?=)\d"), "_p_val_"),
    (re.compile("^((\d+-)?year-old|y\.?o\.?)$"), "_age_"),
    (re.compile("^~?-?\d*[·.]?\d+--?\d*[·.]?\d+$"), "_range_"),
    (re.compile("[a-zA-Z]?(~?[=<>≤≥]|</?=|>/?=)\d?|^(lt|gt|geq|leq)$"), "_ineq_"),
    (re.compile("^~?\d+-fold$"), "_n_fold_"),
    (re.compile("^~?\d+/\d+$|^\d+:\d+$"), "_ratio_"),
    (re.compile("^~?-?\d*[·.]?\d*%$"), "_percent_"),
    (re.compile("^~?\d*(("
                "(kg|\d+g|mg|ug|ng)|"
                "(\d+m|cm|mm|um|nm)|"
                "(\d+l|ml|cl|ul|mol|mmol|nmol|mumol|mo))/?)+$"), "_unit_"),
    # abbreviation starting with letters and containing nums
    (re.compile("^[Rr][Ss]\d+$|"
                "^[Rr]\d+[A-Za-z]$"), "_mutation_"),
    (re.compile("^[a-zA-Z]\w*-?\w*\d+\w*$"), "_abbrev_"),
    # time
    (re.compile("^([jJ]an\.(uary)?|[fF]eb\.(ruary)?|[mM]ar\.(ch)?|"
                "[Aa]pr\.(il)?|[Mm]ay\.|[jJ]un\.(e)?|"
                "[jJ]ul\.(y)?|[aA]ug\.(ust)?|[sS]ep\.(tember)?|"
                "[oO]ct\.(ober)?|[nN]ov\.(ember)?|[dD]ec\.(ember)?)$"), "_month_"),
    (re.compile("^(19|20)\d\d$"), "_year_"),
    # numbers
    (re.compile("^(([Zz]ero(th)?|[Oo]ne|[Tt]wo|[Tt]hree|[Ff]our(th)?|"
                "[Ff]i(ve|fth)|[Ss]ix(th)?|[Ss]even(th)?|[Ee]ight(th)?|"
                "[Nn]in(e|th)|[Tt]en(th)?|[Ee]leven(th)?|"
                "[Tt]went(y|ieth)?|[Tt]hirt(y|ieth)?|[Ff]ort(y|ieth)?|[Ff]ift(y|ieth)?|"
                "[Ss]ixt(y|ieth)?|[Ss]event(y|ieth)?|[Ee]ight(y|ieth)?|[Nn]inet(y|ieth)?|"
                "[Mm]illion(th)?|[Bb]illion(th)?|"
                "[Tt]welv(e|th)|[Hh]undred(th)?|[Tt]housand(th)?|"
                "[Ff]irst|[Ss]econd|[Tt]hird|\d*1st|\d*2nd|\d*3rd|\d+-?th)-?)+$"), "_num_"),
    (re.compile("^~?-?\d+(,\d\d\d)*$"), "_num_"),  # int (+ or -)
    (re.compile("^~?-?((-?\d*[·.]\d+$|^-?\d+[·.]\d*)(\+/-)?)+$"), "_num_"),  # float (+ or -)
    # misc. abbrevs
    (re.compile("^[Vv]\.?[Ss]\.?$|^[Vv]ersus$"), "vs"),
    (re.compile("^[Ii]\.?[Ee]\.?$"), "ie"),
    (re.compile("^[Ee]\.?[Gg]\.?$"), "eg"),
    (re.compile("^[Ii]\.?[Vv]\.?$"), "iv"),
    (re.compile("^[Pp]\.?[Oo]\.?$"), "po")
]


def prenormalize(text):
    """normalize common abbreviations and symbols known to mess with sentence boundary disambiguation"""
    for regex, repl in prenormalize_dict:
        text = regex.sub(repl, text)

    return text


lower_ahead = "(?=[a-z0-9])"
nonw_behind = "(?<=\W)"

prenormalize_dict = [
    # replace ":" or whitespace after section headlines with dots so they will become separate sentences
    # TODO check if this is better or worse (isolate headlines)
    # (re.compile("(AIMS?|BACKGROUNDS?|METHODS?|RESULTS?|CONCLUSIONS?|PATIENTS?|FINDINGS?|FUNDINGS?)" "(:)"), r"\1. "),

    # common abbreviations
    (re.compile(nonw_behind + "[Ee]\.[Gg]\.\s" + lower_ahead), "e g  "),
    (re.compile(nonw_behind + "[Ee][Gg]\.\s" + lower_ahead), "e g "),
    (re.compile(nonw_behind + "[Ii]\.[Ee]\.\s" + lower_ahead), "i e  "),
    (re.compile(nonw_behind + "[Ii][Ee]\.\s" + lower_ahead), "i e "),
    (re.compile(nonw_behind + "[Aa]pprox\.\s" + lower_ahead), "approx  "),
    (re.compile(nonw_behind + "[Nn]o\.\s" + lower_ahead), "no  "),
    (re.compile(nonw_behind + "[Nn]o\.\s" + "(?=\w\d)"), "no  "),  # no. followed by abbreviation (patient no. V123)
    (re.compile(nonw_behind + "[Cc]onf\.\s" + lower_ahead), "conf  "),
    # scientific writing
    (re.compile(nonw_behind + "et al\.\s" + lower_ahead), "et al  "),
    (re.compile(nonw_behind + "[Rr]ef\.\s" + lower_ahead), "ref  "),
    (re.compile(nonw_behind + "[Ff]ig\.\s" + lower_ahead), "fig  "),
    # medical
    (re.compile(nonw_behind + "y\.o\.\s" + lower_ahead), "y o  "),
    (re.compile(nonw_behind + "yo\.\s" + lower_ahead), "y o "),
    (re.compile(nonw_behind + "[Pp]\.o\.\s" + lower_ahead), "p o  "),
    (re.compile(nonw_behind + "[Ii]\.v\.\s" + lower_ahead), "i v  "),
    (re.compile(nonw_behind + "[Qq]\.i\.\d\.\s" + lower_ahead), "q d  "),
    (re.compile(nonw_behind + "[Bb]\.i\.\d\.\s" + lower_ahead), "b i d  "),
    (re.compile(nonw_behind + "[Tt]\.i\.\d\.\s" + lower_ahead), "t i d  "),
    (re.compile(nonw_behind + "[Qq]\.i\.\d\.\s" + lower_ahead), "q i d  "),
    (re.compile(nonw_behind + "J\.\s" + "(?=(Cell|Bio|Med))"), "J  "),  # journal
    # bracket complications
    (re.compile("\.\)."), " )."),
    (re.compile("\.\s\)."), "  )."),
    # multiple dots
    # (re.compile("(\.+\s*\.+)+"), "."),
    # Typos: missing space after dot; only add space if there are at least two letters before and behind
    (re.compile("(?<=[A-Za-z]{2})" + "\." + "(?=[A-Z][a-z])"), ". "),
]


# ----------------------------------------------------------------
# Features
# ----------------------------------------------------------------


def vectorizer(chargrams=(2, 6), min_df_char=0.001, wordgrams=None, min_df_word=0.001, ner=False, rules=True,
               max_df=1.0, binary=False, normalize=True, stats="length"):
    """Return pipeline that concatenates features from word and character n-grams and text stats"""

    return FeatureNamePipeline([
        ("text_normalizer", None if not normalize else TextNormalizer()),
        ("features", FeatureUnion(n_jobs=2,
                                  transformer_list=[
                                      ("wordgrams", None if wordgrams is None else
                                      FeatureNamePipeline([
                                          ("preprocessor", TokenizePreprocessor(rules=rules, ner=ner)),
                                          ("word_tfidf", TfidfVectorizer(
                                                  analyzer='word',
                                                  min_df=min_df_word,  # TODO find reasonable value (5 <= n << 50)
                                                  max_df=max_df,
                                                  tokenizer=helpers.identity,
                                                  preprocessor=None,
                                                  lowercase=False,
                                                  ngram_range=wordgrams,
                                                  binary=binary, norm='l2' if not binary else None,
                                                  use_idf=not binary))
                                      ])),
                                      ("chargrams", None if chargrams is None else
                                      FeatureNamePipeline([
                                          ("normalize_digits", DigitNormalizer()),
                                          ("char_tfidf", TfidfVectorizer(
                                                  analyzer='char',
                                                  min_df=min_df_char,
                                                  max_df=max_df,
                                                  preprocessor=partial(re.compile("[^\w\-=%]+").sub, " "),
                                                  lowercase=True,
                                                  ngram_range=chargrams,
                                                  binary=binary, norm='l2' if not binary else None,
                                                  use_idf=not binary))
                                      ])),
                                      ("stats", None if stats is None else
                                      FeatureNamePipeline([
                                          ("stats", TextLength()),
                                          ("vect", DictVectorizer())
                                      ]))
                                  ]))
    ])


def vectorizer_dx(chargrams=(2, 6), min_df_char=0.001, wordgrams=None, min_df_word=0.001, ner=False, rules=True,
                  max_df=1.0, binary=False, normalize=True, stats="length"):
    """concatenates vectorizer and additional text stats (e.g. sentence position in abstract)

    all args are forwarded to vectorizer as positional arguments"""

    return FeatureUnion(transformer_list=[
        ("text_features", Pipeline([("text_selector", ItemGetter(0)),
                                    ("text_features",
                                     vectorizer(chargrams=chargrams, min_df_char=min_df_char, wordgrams=wordgrams,
                                                min_df_word=min_df_word, ner=ner, rules=rules,
                                                max_df=max_df, binary=binary, normalize=normalize, stats=stats))])),
        ("stats", Pipeline([("stat_features", StatFeatures()),
                            ("vect", DictVectorizer())]))
    ])


def percentile_selector(score_func='chi2', percentile=20):
    """supervised feature selector"""

    funcs = {'chi2'               : chi2,
             'f_classif'          : f_classif,
             'f'                  : f_classif,
             'mutual_info_classif': mutual_info_classif,
             'mutual_info'        : mutual_info_classif,
             'm'                  : mutual_info_classif,
             }

    func = funcs.get(score_func, chi2)

    print("Supervised feature selection:,", percentile, "-th percentile in terms of", func)
    return SelectPercentile(score_func=func, percentile=percentile)


def select_from_l1_svc(C=0.1, tol=1e-3, threshold="0.5*mean"):
    return SelectFromModel(LinearSVC(C=C, penalty="l1", dual=False, tol=tol),
                           prefit=False, threshold=threshold)


def factorization(method='TruncatedSVD', n_components=10):
    print("Unsupervised feature selection: matrix factorization with", method, "(", n_components, "components )")

    sparse = {
        'LatentDirichletAllocation': LatentDirichletAllocation(n_components=n_components,
                                                               n_jobs=-1,
                                                               learning_method='online'),
        'TruncatedSVD'             : FeatureNamePipeline([("selector", TruncatedSVD(n_components)),
                                                          ("normalizer", MinMaxScaler())])
    }

    model = sparse.get(method, None)

    if model is not None:
        return model

    dense = {
        'PCA'           : PCA(n_components),
        'SparsePCA'     : SparsePCA(n_components),
        'FactorAnalysis': FactorAnalysis(n_components)
    }

    model = dense.get(method, None)

    if model is not None:
        return FeatureNamePipeline([("densifier", Densifier()),
                                    ("selector", model),
                                    ("normalizer", MinMaxScaler())])  # TODO Standard or MinMax?

    else:

        return FeatureNamePipeline([("selector", TruncatedSVD(n_components)),
                                    ("normalizer", MinMaxScaler())])


class TextLength(BaseEstimator, TransformerMixin):
    """Extract features from tokenized document for DictVectorizer

    inverse_length: 1/(number of tokens)
    """

    def __init__(self, inv_len=True, inv_tok_cnt=True):
        key_dict = {
            'inv_len'    : inv_len,
            'inv_tok_cnt': inv_tok_cnt,
        }

        self.key_dict = {key: key for key in key_dict if key_dict[key]}

        super(TextLength, self).__init__()

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        for sentence in X:
            yield {
                'inv_len'    : 1.0 / len(sentence)
                if sentence and self.key_dict.get("inv_len") else 1.0,
                'inv_tok_cnt': (1.0 / len(re.split("\s+", sentence)))
                if self.key_dict.get("inv_tok_cnt") else 1.0
            }

    def get_feature_names(self):
        return list(self.key_dict.keys())


class StatFeatures(BaseEstimator, TransformerMixin):
    """Extract metadata features from document record for a DictVectorizer"""

    def __init__(self, pos=True, title_words=True):
        key_dict = {
            'pos'        : pos,
            'title_words': title_words,
            'zero'       : 'zero'  # TODO delete fake feature
        }

        self.key_dict = {key: key for key in key_dict if key_dict[key]}

        super(StatFeatures, self).__init__()

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        # text, pos, title = [0, 1, 2]
        for x in X:
            yield {
                'pos'        : float(x[1])
                if self.key_dict.get("pos") else 1.0,
                'title_words': self.inverse_matching_ngrams(x[2], x[0], ngram=6)
                if self.key_dict.get("title_words") else 1.0,
                'zero'       : 0  # TODO delete fake feature
            }

    def get_feature_names(self):
        return list(self.key_dict.keys())

    def inverse_matching_ngrams(self, src, txt, ngram=6):
        count = 0
        if type(src) == str:
            for ngram in helpers.ngrams(src, ngram):
                if ngram in txt:
                    count += 1
        return 1.0 if count == 0 else 1.0 / count


# ----------------------------------------------------------------
# Mapping tokens to dictionary equivalents
# ----------------------------------------------------------------

class DictReplacer(object):
    def __init__(self, word_map):
        self.word_map = word_map

    def replace(self, word):
        return self.word_map.get(word, word)

    def replace_all(self, words):
        return [self.replace(w) for w in words]


class HypernymMapper(DictReplacer):
    def __init__(self):
        dictionary = self.load_hypernyms()
        super(HypernymMapper, self).__init__(dictionary)

    def load_hypernyms(self):
        """read hypernym dict from disk or build from tsv files"""
        try:
            with open(file_path("./pickles/hypernyms.pickle"), "rb") as f:
                hypernyms = pickle.load(f)
                # print("Loaded hypernyms from disk.")
        except IOError:
            print("Building hypernym resources...")
            hypernyms = self.build_hypernym_dict()
            with open(file_path("./pickles/hypernyms.pickle"), "wb") as f:
                pickle.dump(hypernyms, f)
                print("Built hypernym dict and wrote to disk.")
        return hypernyms

    def build_hypernym_dict(self):
        concepts = ["chemical", "gene", "mutation"]  # "disease"

        with multi.Pool(min(multi.cpu_count(), len(concepts))) as p:
            dicts = list(p.map(self.make_hypernym_entries, concepts))

        dictionary = dicts[0].copy()
        for d in dicts[1:]:
            dictionary.update(d)

        return dictionary

    def make_hypernym_entries(self, hypernym):
        entries = {}
        source = pd.read_csv(file_path("./resources/" + hypernym + "2pubtator.csv"),
                             sep='\t', dtype=str)

        with open(file_path("./resources/common_words.txt"), "r") as cw:
            common_words = set(cw.read().split("\n") + stopwords.words('english'))

        # only single words and no URLs etc
        illegal_substrs = re.compile("\s|\\\\|\.gov|\.org|\.com|http|www|^n=")

        for line in source["Mentions"]:
            for word in str(line).split("|"):
                # only include single words not appearing in normal language
                if not (illegal_substrs.findall(word)
                        or word in common_words
                        or word in entries
                        or len(word) < 4
                        or wordnet.synsets(word)):
                    entries[word] = "_" + hypernym + "_"

        return entries


# ----------------------------------------------------------------
# helpers
# ----------------------------------------------------------------


class identitySelector(BaseEstimator, TransformerMixin):
    """feature selector that does nothing"""

    def __init__(self):
        print("Feature selection: None")
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class ItemGetter(BaseEstimator, TransformerMixin):
    """Pseudo-transformer to get subsets (param idx: int or list)) of samples for subsequent transformers"""

    def __init__(self, idx):
        self.idx = idx
        return

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        return np.array(X)[:, self.idx]


class Densifier(BaseEstimator, TransformerMixin):
    """Makes sparse matrices dense for subsequent pipeline steps"""

    def __init__(self):
        return

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        return helpers.densify(X)


class FeatureNamePipeline(Pipeline):
    """Wrapper for Pipeline able to return last step's feature names"""

    def get_feature_names(self):
        return self._final_estimator.get_feature_names()
