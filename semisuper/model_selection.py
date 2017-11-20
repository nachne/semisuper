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
