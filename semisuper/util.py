import warnings

warnings.filterwarnings("ignore")

import itertools
import sys
import traceback
# from _config import *

# sklearn
from scipy.stats import randint as sp_randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression, SGDClassifier, Lasso, ElasticNet
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import cohen_kappa_score, accuracy_score, average_precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectPercentile, chi2
from collections import Counter
import time
import pandas as pd

from Bio import Entrez

Entrez.email = "seva@informatik.hu-berlin.de"


# /home/shared/embeddings/pubmed_char.d200.bin

class getBestModel:
    def __init__(self, modelFolder, modelName, data, labels, n_jobs, testData=None, testLabels=None):
        # self.getFastTextWE()
        self.trainingData = []
        self.trainingDataLabels = []
        self.chapters = []
        self.blocks = []
        self.data = data
        self.labels = labels

        self.testData = testData
        self.testLabels = testLabels

        self.n_iter_search = 20
        self.n_jobs = n_jobs
        self.randIntMax = 1000
        self.modelFolder = modelFolder
        self.modelName = modelName

        self.estimators = [
            RandomForestClassifier(),
            # DecisionTreeClassifier(),
            LogisticRegression(),
            # SGDClassifier(),
            SVC(),
            MultinomialNB(),
            # MLPClassifier()
        ]
        self.names = [
            "RandomForestClassifier",
            # "DecisionTreeClassifier",
            "LogisticRegression",
            # "SGDClassifier",
            "SVM_SVC",
            "MultinomialNB",
            # "MLPClassifier"
        ]

        self.estimatRandomSearchParameters = {
            "LogisticRegression"    : {
                'C'           : sp_randint(1, self.randIntMax),
                'solver'      : ['newton-cg', 'lbfgs', 'liblinear'],
                'class_weight': ['balanced']
            },
            "SVM_SVC"               : {
                'C'           : sp_randint(1, self.randIntMax),
                'kernel'      : ['linear', 'poly', 'rbf', 'sigmoid'],
                'class_weight': ['balanced'],
                'probability' : [True]
            },
            "SVM_LinearSVC"         : {
                'C'           : sp_randint(1, self.randIntMax),
                'class_weight': ['balanced']
            },
            "DecisionTreeClassifier": {
                "criterion"   : ["gini", "entropy"],
                "splitter"    : ["best", "random"],
                'max_depth'   : sp_randint(1, 1000),
                'class_weight': ['balanced']
            },
            "RandomForestClassifier": {
                'n_estimators': sp_randint(1, self.randIntMax),
                "criterion"   : ["gini", "entropy"],
                'max_depth'   : sp_randint(1, self.randIntMax),
                'class_weight': ['balanced']
            },
            "KNeighbors"            : {
                'n_neighbors' : sp_randint(1, 40),
                'weights'     : ['uniform', 'distance'],
                'algorithm'   : ['auto'],
                'leaf_size'   : sp_randint(1, self.randIntMax),
                'class_weight': ['balanced']
            },
            "SGDClassifier"         : {
                'loss'         : ['hinge', 'log', 'modified_huber', 'squared_hinge'],
                'class_weight' : ['balanced'],
                'penalty'      : ['l2', 'l1', 'elasticnet'],
                'learning_rate': ['optimal', 'invscaling'],
                'eta0'         : uniform(0.01, 0.00001)
            },
            "MultinomialNB"         : {
                'alpha'    : uniform(0, 1),
                'fit_prior': [True],
            },
            "LinearSVC"             : {
                'C'   : uniform(0, 1),
                'loss': ['hinge', 'squared_hinge']
            },
            "Lasso"                 : {
                'alpha'        : uniform(0, 1),
                'fit_intercept': [True],
                'normalize'    : [True, False],
                'max_iter'     : sp_randint(1, self.randIntMax)
            },
            "ElasticNet"            : {
                'alpha'   : uniform(0, 1),
                'l1_ratio': uniform(0, 1)
            },
            "MLPClassifier"         : {
                'activation'   : ['identity', 'logistic', 'tanh', 'relu'],
                'solver'       : ['lbfgs', 'sgd', 'adam'],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'max_iter'     : [100000]
            }
        }

    def getFastTextWE(self):
        self.embeddingModels = {}
        for k, v in langWE.iteritems():
            if k in LANG_SUPPORT:
                print("Loading FastText embeddings for {}".format(k))
                try:
                    self.embeddingModels[k] = fasttext.load_model(v)
                except Exception as e:
                    print(e)

    def createFeatureVector(self, dataSet):

        print
        'Extracting features'
        dummyDS = list()

        for i, row in enumerate(dataSet):
            dummyDS.append(list(itertools.chain(*[self.embeddingModels['EN'][row], self.embeddingModels['FR'][row]])))
            # dummyDS.append(np.average([self.embeddingModels['EN'][row], self.embeddingModels['FR'][row] ]))
        return dummyDS

    def buildVocabulary(self, ngrams):
        # =======================================================================
        # create feautres per label, count disticnt features per label, concatenate to a feature vocab and use that to perform TfidfVectorizer 
        # =======================================================================

        ct = CountVectorizer(ngram_range=ngrams, lowercase=True, strip_accents='unicode', token_pattern=u'\S+[^.,!?\s]')

        yes = []
        no = []

        for a, b in zip(self.data, self.labels):
            if b == 1:
                yes.append(a)

            if b == 0:
                no.append(a)

        yesCT = ct.fit(yes).vocabulary_.keys()
        noCT = ct.fit(no).vocabulary_.keys()

        onlyYes = list(set(yesCT) - set(noCT))
        onlyNo = list(set(noCT) - set(yesCT))

        return onlyNo + onlyYes

    def prepareTrainTest(self, ngramRange, trainData, testData, trainLabels, max_df_freq, analyzerLevel='word',
                         featureSelect=False, vocab=None):

        tfidfVect = TfidfVectorizer(ngram_range=ngramRange, analyzer=analyzerLevel, norm='l2', decode_error='replace',
                                    max_df=max_df_freq, sublinear_tf=True,
                                    lowercase=True, strip_accents='unicode', token_pattern=u'\S+[^.,!?\s]',
                                    vocabulary=vocab)
        transformedTrainData = tfidfVect.fit_transform(trainData)
        transformedTestData = tfidfVect.transform(testData)

        # def featureSelection(self, trainData, trainLabels, testData):
        ch2 = None
        if featureSelect:
            print
            "Selecting best features"
            ch2 = SelectPercentile(chi2, 20)
            transformedTrainData = ch2.fit_transform(transformedTrainData, trainLabels)
            transformedTestData = ch2.transform(transformedTestData)

        # print 'Transformed train data set feature space size:\tTrain {}\t\t Test{}'.format(transformedTrainData.shape, transformedTestData.shape)
        return transformedTrainData, transformedTestData, tfidfVect, ch2

    def getBestModel(self, resultsFilename, relevantClass=None, max_df_frequency=[1.0]):

        results = []
        resultsRelevant = []
        ct = (time.strftime("%Y_%m_%d"))

        bestResult = {
            'word': {
                'best'          : 0,
                'random_search_': None,
                'best_ngram'    : None,
                'best_chi2'     : None,
                'model'         : None,
                'analyzer'      : [(1, 2), (1, 3), (1, 4)]
            },
            'char': {
                'best'          : 0,
                'p'             : 0,
                'f'             : 0,
                'r'             : 0,
                'random_search_': None,
                'best_ngram'    : None,
                'best_chi2'     : None,
                'model'         : None,
                'analyzer'      : [(1, 4), (2, 4), (2, 5), (2, 6)]
            }
        }

        TrainData, TestData, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.2, random_state=42,
                                                                stratify=self.labels)
        print
        'Splitting training data in dev/eval sets:\tDev {}\tEval{}'.format(Counter(y_train), Counter(y_test))

        for k, v in bestResult.iteritems():
            for ngram in v['analyzer']:
                for max_freq in max_df_frequency:
                    X_train, X_test, tfidf, ch2 = self.prepareTrainTest(ngram, TrainData, TestData, y_train, max_freq,
                                                                        k)
                    for i, model in enumerate(self.estimators):
                        try:

                            random_search = RandomizedSearchCV(model,
                                                               param_distributions=self.estimatRandomSearchParameters[
                                                                   self.names[i]],
                                                               n_iter=self.n_iter_search,
                                                               n_jobs=self.n_jobs,
                                                               pre_dispatch='n_jobs',
                                                               cv=10,
                                                               scoring='f1_macro',
                                                               verbose=0)

                            random_search.fit(X_train, y_train)

                            # ===============================================================
                            # EVALUATE ON EVAL DS
                            # ===============================================================
                            y_predicted = random_search.best_estimator_.predict(X_test)
                            p, r, f, s = precision_recall_fscore_support(y_test, y_predicted, pos_label=None,
                                                                         average='macro')
                            acc = accuracy_score(y_test, y_predicted)
                            print
                            '{},{},{}, ngram({},{})'.format(self.names[i], k, max_freq, ngram[0], ngram[1])
                            print
                            '\tEVAL:\t\t\t', p, r, f, acc
                            # print classification_report(y_test, y_predicted)
                            results.append([self.names[i],
                                            k,
                                            '({},{})'.format(ngram[0], ngram[1]),
                                            'EVAL',
                                            max_freq,
                                            p, r, f, acc])

                            # ===============================================================
                            # EVALUATE ON TEST DS
                            # ===============================================================
                            if self.testData:
                                if ch2:
                                    transformedTestData = ch2.transform(tfidf.transform(self.testData))
                                else:
                                    transformedTestData = tfidf.transform(self.testData)

                                y_predicted_test = random_search.best_estimator_.predict(transformedTestData)
                                p, r, f, s = precision_recall_fscore_support(self.testLabels, y_predicted_test,
                                                                             average='macro')
                                acc = accuracy_score(self.testLabels, y_predicted_test)
                                print
                                '\tTEST:\t\t\t', p, r, f, acc
                                results.append([self.names[i],
                                                k,
                                                '({},{})'.format(ngram[0], ngram[1]),
                                                'TEST',
                                                max_freq,
                                                p, r, f, acc])

                                # print classification_report(self.testLabels, y_predicted_test)

                            if relevantClass:
                                pr, rr, fr, sr = precision_recall_fscore_support(self.testLabels, y_predicted_test,
                                                                                 pos_label=relevantClass,
                                                                                 average='binary')
                                acc = acc * fr
                                print
                                '\tTEST ({}):\t'.format(relevantClass), pr, rr, fr, acc
                                print
                                '------------------'

                                resultsRelevant.append([self.names[i],
                                                        k,
                                                        '({},{})'.format(ngram[0], ngram[1]),
                                                        max_freq,
                                                        pr, rr, fr, acc])

                            # ===========================================================
                            # create best model
                            # ===========================================================
                            if acc > bestResult[k]['best']:
                                bestResult[k]['model'] = self.names[i]
                                bestResult[k]['best'] = acc
                                bestResult[k]['random_search_'] = random_search
                                bestResult[k]['best_ngram'] = tfidf
                                bestResult[k]['best_chi2'] = ch2

                        except Exception as e:
                            print
                            'error:\t', e
                            traceback.print_exc()
                            sys.exit()

        header = ['Model', 'NGram_Feature', 'NGramRange', 'OnDataset', 'TFIDF_CutOff', 'P', 'R', 'F1', 'Accuracy']
        df = pd.DataFrame(results)
        df.to_csv(RESULTS + '{}_{}.csv'.format(ct, resultsFilename), index=False, header=header, sep=";")

        # results on test data set
        if resultsRelevant:
            header = ['Model', 'NGram_Feature', 'NGramRange', 'TFIDF_CutOff', 'P', 'R', 'F1', 'Acc(All)*FR(Relevant)']
            df = pd.DataFrame(resultsRelevant)
            df.to_csv(RESULTS + '{}_{}_TEST_DS.csv'.format(ct, resultsFilename), index=False, header=header, sep=";")

        try:
            for k in bestResult:

                # ===============================================================
                # CREATE MODEL BASED ON BEST 
                # ===============================================================
                print('Fitting best model on all data for {} feature level with {}'.format(k, bestResult[k]['model']))
                tfidfVect = bestResult[k]['best_ngram']
                ch2 = bestResult[k]['best_chi2']

                print(tfidfVect)
                print(ch2)

                if ch2:
                    transformedData = ch2.fit_transform(tfidfVect.fit_transform(self.data), self.labels)
                    transformedTestData = ch2.fit_transform(tfidfVect.fit_transform(self.testData), self.testLabels)
                else:
                    transformedData = tfidfVect.fit_transform(self.data)
                    transformedTestData = tfidfVect.transform(self.testData)

                classModel = bestResult[k]['random_search_'].best_estimator_.fit(transformedData, self.labels)
                joblib.dump(classModel, '{}{}_{}.classifier'.format(self.modelFolder, self.modelName, k))
                joblib.dump(tfidfVect, '{}{}_{}.vectorizer'.format(self.modelFolder, self.modelName, k))
                joblib.dump(ch2, '{}{}_{}.featureSelector'.format(self.modelFolder, self.modelName, k))

                # ===============================================================
                # PERFORMANCE FO MODEL ON TEST DATA
                # ===============================================================
                y_predicted_test = classModel.predict(transformedTestData)
                print(classification_report(self.testLabels, y_predicted_test))

        except Exception as e:
            print('error:\t', e)
            traceback.print_exc()
            sys.exit()
