from semisuper.util import *

# ===============================================================================
# TRAIN AND EVALUATE MODELS
# ===============================================================================
testDS = pickle.load(open('datasets/testDataSet.p', 'rb'))
testlabels = [x[0].strip() for x in testDS]
testData = [x[1].strip() for x in testDS]

trainDatasets = {
    'clinicalRelevance_OncoKB_balanced': trainDataSet,
    'clinicalRelevance_OncoKB_original': trainDataSet_OncoKB,
    'clinicalRelevance_OncoKB_CIViC'   : trainDataSet_combined
}

for k, v in trainDatasets.iteritems():
    labels = [x[0] for x in v]
    data = [x[1] for x in v]
    print
    "Training on {} with {} documents distrbiuted as {}".format(k, len(trainDataSet), Counter(labels))
    # print "Ration of train data:", Counter(labels)

    print
    "Testing with {} documents with distribution of classes {}".format(len(testDataSet), Counter(testlabels))
    # print "Ration of test data:", Counter(testlabels)

    bm = getBestModel('models/clinical/', k, data, labels, 70, testData, testlabels)
    bm.getBestModel(k, 'Relevant')
