import string
import re
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk import TreebankWordTokenizer, WhitespaceTokenizer, RegexpTokenizer, wordpunct_tokenize
from nltk import sent_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk import WordNetLemmatizer
from dict_matchers import HypernymMapper


class BasicPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, punct=None, lower=True, strip=True, lemmatize=False):
        self.lower = lower
        self.strip = strip
        self.punct = punct or set(string.punctuation)
        self.lemma = lemmatize

        self.splitters = re.compile("[-/.,|]")

        self.dict_mapper = HypernymMapper()
        self.tokenizer = TreebankWordTokenizer()
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [", ".join(doc) for doc in X]

    def transform(self, X):
        return [self.representation(sentence) for sentence in X]

    def representation(self, sentence):
        tokens_tags = list(self.tokenize(sentence))
        return [t for (t, p) in tokens_tags]

    def tokenize(self, sentence):
        """break sentence into pos-tagged tokens; normalize and split on hyphens"""

        # extremely short sentences shall be ignored by next steps
        if len(sentence) < 6:
            return []

        for token, tag in pos_tag(self.tokenizer.tokenize(sentence)):
            # Apply preprocessing to the token
            token_nrm = self.normalize_token(token, tag)

            subtokens = [self.normalize_token(t, tag) for t in self.splitters.split(token_nrm)]

            for subtoken in subtokens:

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                yield subtoken, tag

    def normalize_token(self, token, tag):
        # Apply preprocessing to the token
        token = token.strip() if self.strip else token
        token = token.strip('*') if self.strip else token
        token = token.strip('.') if self.strip else token

        token = self.dict_mapper.replace(token)
        token = map_regex_concepts(token)

        token = token.lower() if self.lower else token

        if self.lemma:
            token = self.lemmatize(token, tag)
        return token

    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)


def map_regex_concepts(token):
    """replaces abbreviations matching simple REs, e.g. for numbers, percentages, gene names, by class tokens"""

    for regex in regex_concept_dict.keys():
        if regex.findall(token):
            return regex_concept_dict[regex]

    return token


# TODO dict ist nicht gut, weil Reihenfolge (z.B. Prozent kommt nie durch)
regex_concept_dict = {  # number-related concepts
    re.compile("^[Pp]([=<>≤≥]|</?=|>/?=)\d")                                : "_p_val_",
    re.compile("^((\d+-)?year-old|y\.?o\.?)$")                              : "_age_",
    re.compile("^~?-?\d*[·.]?\d+--?\d*[·.]?\d+$")                           : "_range_",
    re.compile("[a-zA-Z]?(~?[=<>≤≥]|</?=|>/?=)\d|^(lt|gt|geq|leq)$")        : "_ineq_",
    re.compile("^~?\d+-fold$")                                              : "_n_fold_",
    re.compile("^~?\d+/\d+$|^\d+:\d+$")                                     : "_ratio_",
    re.compile("^~?-?\d*[·.]?\d*%$")                                        : "_percent_",
    re.compile("^~?\d*(("
               "(kg|g|mg|ug|ng)|"
               "(m|cm|mm|um|nm)|"
               "(l|ml|cl|ul|mol|mmol|nmol|mumol|mo))/?)+$")                 : "_unit_",
    # abbreviation starting with letters and containing nums
    re.compile("^rs\d+$")                                                   : "_mutation_",
    re.compile("^([a-zA-Z]+-?\w*\d+)+")                                     : "_abbrev_",
    # time
    re.compile("^([jJ]an\.(uary)?|[fF]eb\.(ruary)?|[mM]ar\.(ch)?|"
               "[Aa]pr\.(il)?|[Mm]ay\.|[jJ]un\.(e)?|"
               "[jJ]ul\.(y)?|[aA]ug\.(ust)?|[sS]ep\.(tember)?|"
               "[oO]ct\.(ober)?|[nN]ov\.(ember)?|[dD]ec\.(ember)?)$")       : "_month_",
    re.compile("^(19|20)\d\d$")                                             : "_year_",
    # numbers
    re.compile("^([Zz]ero(th)?|[Oo]ne|[Tt]wo|[Tt]hree|[Ff]our(th)?|"
               "[Ff]i(ve|fth)|[Ss]ix(th)?|[Ss]even(th)?|[Ee]ight(th)?|"
               "[Nn]in(e|th)|[Tt]en(th)?|[Ee]leven(th)?|"
               "[Tt]welv(e|th)|[Hh]undred(th)?|[Tt]housand(th)?|"
               "[Ff]irst|[Ss]econd|[Tt]hird|\d*1st|\d*2nd|\d*3rd|\d+-?th)$"): "_num_",
    re.compile("^~?-?\d+(,\d+)?$")                                          : "_num_",  # int (+ or -)
    re.compile("^~?-?((-?\d*[·.]\d+$|^-?\d+[·.]\d*)(\+/-)?)+$")               : "_num_",  # float (+ or -)
    # misc. abbrevs
    re.compile("^[Vv]\.?[Ss]\.?$|^[Vv]ersus$")                              : "vs",
    re.compile("^[Ii]\.?[Ee]\.?$")                                          : "ie",
    re.compile("^[Ee]\.?[Gg]\.?$")                                          : "eg",
    re.compile("^[Ii]\.?[Vv]\.?$")                                          : "iv",
    re.compile("^[Pp]\.?[Oo]\.?$")                                          : "po"
}


def sentence_tokenize(text):
    """tokenize text into sentences after simple normalization"""
    return sent_tokenize(prenormalize(text))


# TODO: detect and isolate headlines?
# TODO: switch from dict to hardcoded to capture following lower-cases?
def prenormalize(text):
    """normalize common abbreviations and symbols known to mess with sentence boundary disambiguation"""
    for regex in prenormalize_dict.keys():
        text = regex.sub(prenormalize_dict[regex], text)

    return text


prenormalize_dict = {
    # usual abbreviations
    re.compile("\W[Ee]\.[Gg]\.\s") : " eg ",
    re.compile("\W[Ii]\.[Ee]\.\s") : " ie ",
    re.compile("\W[Aa]pprox\.\s")  : " approx ",
    re.compile("\W[Nn]o\.\s")      : " no ",
    # scientific writing
    re.compile("\Wet al\.\s")      : " et al ",
    re.compile("\W[Rr]ef\.\s")     : " ref ",
    re.compile("\W[Ff]ig\.\s")     : " fig ",
    # medical
    re.compile("\Wy\.?o\.\s")      : " year-old ",
    re.compile("\W[Pp]\.o\.\s")    : " po ",
    re.compile("\W[Ii]\.v\.\s")    : " iv ",
    re.compile("\W[Qq]\.i\.\d\.\s"): " qd ",
    re.compile("\W[Bb]\.i\.\d\.\s"): " bid ",
    re.compile("\W[Tt]\.i\.\d\.\s"): " tid ",
    re.compile("\W[Qq]\.i\.\d\.\s"): " qid ",
    # bracket complications
    re.compile("\.\s*\).")         : ").",
    # double dots
    re.compile("(\.\s*\.)+")       : ".",
    re.compile("wild-type")        : "wild type"
}


class FeatureNamePipeline(Pipeline):
    def get_feature_names(self):
        return self._final_estimator.get_feature_names()


# TODO make piped class from this and DictVectorize
class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from tokenized document for DictVectorizer

    inverse_length: 1/(number of tokens)
    """

    key_dict = {'inverse_length': 'inverse_length'}

    def fit(self, X=None, y=None):
        return self

    def transform(self, token_lists):
        for tl in token_lists:
            yield {'inverse_length': (1.0 / len(tl) if tl else 1.0)}

    def get_feature_names(self):
        return list(self.key_dict.keys())
