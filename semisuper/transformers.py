import os.path
import re
import string
from functools import partial

import numpy as np
from geniatagger import GeniaTagger
from nltk import TreebankWordTokenizer
from nltk import sent_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, PCA, SparsePCA, FactorAnalysis
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectPercentile, f_classif, mutual_info_classif, SelectFromModel
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from unidecode import unidecode

from semisuper import helpers

# ----------------------------------------------------------------
# Required globals
# ----------------------------------------------------------------


MIN_LEN = 8


# ----------------------------------------------------------------
# Tokenization
# ----------------------------------------------------------------

class TokenizePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, rules=True, genia_opts=None):

        self.punct = set(string.punctuation).difference(set('%='))

        if genia_opts:
            self.genia = True

            self.chunks = False # TODO decide whether to drop chunks altogether

            self.pos = genia_opts["pos"] if not self.chunks else False
            self.ner = genia_opts["ner"]

            self.create_global_tagger()
        else:
            self.genia = self.pos = self.ner = self.chunks = False

        self.rules = rules

        self.splitters = re.compile("[-/.,|<>]")
        self.tokenizer = None if genia_opts else TreebankWordTokenizer()

    def fit(self, X=None, y=None):
        return self

    def inverse_transform(self, X):
        return [", ".join(doc) for doc in X]

    def transform(self, X):
        if self.genia:
            try:
                global tagger
                tagger.parse("test")
            except:
                self.create_global_tagger()
            return [self.genia_representation(sentence) for sentence in X]
        else:
            return [self.token_representation(sentence) for sentence in X]

    def genia_representation(self, sentence):
        if self.chunks:
            return list(self.genia_base_chunks(sentence))
        if self.ner:
            return list(self.genia_tokenize_replace_ne(sentence))
        else:
            return list(self.genia_tokenize(sentence))

    def genia_base_chunks(self, sentence):
        chunk_tmp = ""

        for token, base, pos, chunk, ne in tagger.parse(sentence):
            if chunk[0] == 'I':
                if not all(char in self.punct for char in base):
                    chunk_tmp += " " + self.normalize_token(base)
            else:
                if chunk_tmp:
                    yield chunk_tmp
                if not all(char in self.punct for char in base):
                    chunk_tmp = self.normalize_token(base)
                else:
                    chunk_tmp = ""
        if chunk_tmp:
            yield chunk_tmp

    def genia_tokenize_replace_ne(self, sentence):
        for token, base, pos, chunk, ne in tagger.parse(sentence):
            if ne[0] == 'I':
                pass
            elif ne[0] == 'B':
                yield ("_" + ne[2:] + "_" + (" " + pos if self.pos else ""))
            else:
                if not all(char in self.punct for char in token):
                    yield (self.normalize_token(token) + (" " + pos if self.pos else ""))

    def genia_tokenize(self, sentence):
        for token, base, pos, chunk, ne in tagger.parse(sentence):
            # Apply preprocessing to the token
            token_nrm = self.normalize_token(token)

            # If punctuation, ignore token and continue
            if all(char in self.punct for char in token_nrm):
                continue
            if self.pos:
                yield token_nrm + " " + pos
            else:
                yield token_nrm

    def token_representation(self, sentence):
        tokens = list(self.tokenize(sentence))
        return tokens if tokens else ["_empty_sentence_"]

    def tokenize(self, sentence):
        """break sentence into pos-tagged tokens; normalize and split on hyphens"""

        # extremely short sentences shall be ignored by next steps
        if len(sentence) < MIN_LEN:
            return []

        for token in self.tokenizer.tokenize(sentence):
            # Apply preprocessing to the token
            token_nrm = self.normalize_token(token)
            subtokens = [self.normalize_token(t) for t in self.splitters.split(token_nrm)]

            for subtoken in subtokens:
                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue
                yield subtoken

    def normalize_token(self, token):
        # Apply preprocessing to the token
        token = token.lower().strip().strip('*').strip('.')

        if self.rules:
            token = map_regex_concepts(token)

        return token

    def create_global_tagger(self):
        """hacky solution for using GENIA tagger and still being picklable"""
        # TODO this is ugly, try to find another solution
        print("Instantiating new GENIA tagger")
        global tagger
        tagger = new_genia_tagger()
        return


# ----------------------------------------------------------------
# Sentence splitting, Normalization, RegEx based replacement
# ----------------------------------------------------------------

def sentence_tokenize(text):
    """tokenize text into sentences after simple normalization"""
    return sent_tokenize(prenormalize(text))


class TextNormalizer(BaseEstimator, TransformerMixin):
    """replaces all non-ASCII characters by approximations, all numbers by 1"""

    def __init__(self):
        return

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
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
nonword_behind = "(?<=\W)"

prenormalize_dict = [
    # common abbreviations
    (re.compile(nonword_behind + "[Cc]a\.\s" + lower_ahead), "ca  "),
    (re.compile(nonword_behind + "[Ee]\.[Gg]\.\s" + lower_ahead), "e g  "),
    (re.compile(nonword_behind + "[Ee][Gg]\.\s" + lower_ahead), "e g "),
    (re.compile(nonword_behind + "[Ii]\.[Ee]\.\s" + lower_ahead), "i e  "),
    (re.compile(nonword_behind + "[Ii][Ee]\.\s" + lower_ahead), "i e "),
    (re.compile(nonword_behind + "[Aa]pprox\.\s" + lower_ahead), "approx  "),
    (re.compile(nonword_behind + "[Nn]o\.\s" + lower_ahead), "no  "),
    (re.compile(nonword_behind + "[Nn]o\.\s" + "(?=\w\d)"), "no  "),  # no. followed by abbreviation (patient no. V123)
    (re.compile(nonword_behind + "[Cc]onf\.\s" + lower_ahead), "conf  "),
    # scientific writing
    (re.compile(nonword_behind + "et al\.\s" + lower_ahead), "et al  "),
    (re.compile(nonword_behind + "[Rr]ef\.\s" + lower_ahead), "ref  "),
    (re.compile(nonword_behind + "[Ff]ig\.\s" + lower_ahead), "fig  "),
    # medical
    (re.compile(nonword_behind + "y\.o\.\s" + lower_ahead), "y o  "),
    (re.compile(nonword_behind + "yo\.\s" + lower_ahead), "y o "),
    (re.compile(nonword_behind + "[Pp]\.o\.\s" + lower_ahead), "p o  "),
    (re.compile(nonword_behind + "[Ii]\.v\.\s" + lower_ahead), "i v  "),
    (re.compile(nonword_behind + "[Qq]\.i\.\d\.\s" + lower_ahead), "q d  "),
    (re.compile(nonword_behind + "[Bb]\.i\.\d\.\s" + lower_ahead), "b i d  "),
    (re.compile(nonword_behind + "[Tt]\.i\.\d\.\s" + lower_ahead), "t i d  "),
    (re.compile(nonword_behind + "[Qq]\.i\.\d\.\s" + lower_ahead), "q i d  "),
    (re.compile(nonword_behind + "J\.\s" + "(?=(Cell|Bio|Med))"), "J  "),  # journal
    # bracket complications
    (re.compile("\.\)."), " )."),
    (re.compile("\.\s\)."), "  )."),
    # multiple dots
    # (re.compile("(\.+\s*\.+)+"), "."),
    # Typos: missing space after dot; only add space if there are at least two letters before and behind
    (re.compile("(?<=[A-Za-z]{2})" + "\." + "(?=[A-Z][a-z])"), ". "),
    # whitespace
    (re.compile("\s"), " "),
]


# ----------------------------------------------------------------
# Features
# ----------------------------------------------------------------


def vectorizer(chargrams=(2, 6), min_df_char=0.001, wordgrams=(1, 3), min_df_word=0.001, genia_opts=None, rules=True,
               max_df=1.0, binary=False, normalize=True, stats="length"):
    """Return pipeline that concatenates features from word and character n-grams and text stats"""

    return Pipeline([
        ("text_normalizer", None if not normalize else TextNormalizer()),
        ("features", FeatureUnion(n_jobs=2,
                                  transformer_list=[
                                      ("wordgrams", None if wordgrams is None else
                                      Pipeline([
                                          ("preprocessor", TokenizePreprocessor(rules=rules, genia_opts=genia_opts)),
                                          ("word_tfidf", TfidfVectorizer(
                                                  analyzer='word',
                                                  min_df=min_df_word,
                                                  max_df=max_df,
                                                  tokenizer=helpers.identity,
                                                  preprocessor=None,
                                                  lowercase=False,
                                                  ngram_range=wordgrams,
                                                  binary=binary, norm='l2' if not binary else None,
                                                  use_idf=not binary))
                                      ])),
                                      ("chargrams", None if chargrams is None else
                                      Pipeline([
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
                                      Pipeline([
                                          ("stats", TextLength()),
                                          ("vect", DictVectorizer())
                                      ]))
                                  ]))
    ])


def vectorizer_dx(chargrams=(2, 6), min_df_char=0.001, wordgrams=None, min_df_word=0.001, genia_opts=None, rules=True,
                  max_df=1.0, binary=False, normalize=True, stats="length"):
    """concatenates vectorizer and additional text stats (e.g. sentence position in abstract)

    all args are forwarded to vectorizer as positional arguments"""

    return FeatureUnion(transformer_list=[
        ("text_features", Pipeline([("text_selector", ItemGetter(0)),
                                    ("text_features",
                                     vectorizer(chargrams=chargrams, min_df_char=min_df_char, wordgrams=wordgrams,
                                                min_df_word=min_df_word, genia_opts=genia_opts, rules=rules, max_df=max_df,
                                                binary=binary, normalize=normalize, stats=stats))])),
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
        'TruncatedSVD'             : Pipeline([("selector", TruncatedSVD(n_components)),
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
        return Pipeline([("densifier", Densifier()),
                         ("selector", model),
                         ("normalizer", MinMaxScaler())])

    else:

        return Pipeline([("selector", TruncatedSVD(n_components)),
                         ("normalizer", MinMaxScaler())])


class TextLength(BaseEstimator, TransformerMixin):
    """Extract features from tokenized document for DictVectorizer

    inverse_length: 1/(number of tokens)
    """

    def __init__(self, inv_len=True, inv_tok_cnt=False):
        self.inv_len = inv_len
        self.inv_tok_cnt = inv_tok_cnt

        super(TextLength, self).__init__()

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        for sentence in X:
            yield {
                'inv_len'    : 1.0 / len(sentence) if sentence and self.inv_len else 1.0,
                'inv_tok_cnt': (1.0 / len(re.split("\s+", sentence))) if self.inv_tok_cnt else 1.0,
            }


class StatFeatures(BaseEstimator, TransformerMixin):
    """Extract metadata features from document record for a DictVectorizer"""

    def __init__(self, pos=True, title_words=True):
        self.pos= pos
        self.title_words= title_words
        super(StatFeatures, self).__init__()

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        # text, pos, title = [0, 1, 2]
        for x in X:
            yield {
                'pos'        : float(x[1]) if self.pos else 1.0,
                'title_words': self.matching_ngrams_ratio(x[2], x[0], ngram=6) if self.title_words else 1.0,
            }

    def matching_ngrams_ratio(self, src, txt, ngram=6):
        count = 0
        max_val = 0

        if type(src) == str:
            ngrams = helpers.ngrams(src, ngram)
            max_val = len(ngrams)

            for ngram in ngrams:
                if ngram in txt:
                    count += 1
        return 1.0 if count == 0 else max_val / count


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


def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)


def new_genia_tagger():
    return GeniaTagger(file_path("./resources/geniatagger-3.0.2/geniatagger"))

