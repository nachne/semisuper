import preprocessors
import dummy_pipeline
import pickle
import loaders
import random
import re
import pandas as pd

civic, abstracts = loaders.sentences_civic_abstracts()

print("CIViC sentences:", len(civic))
print("abstract sentences:", len(abstracts))

piboso = loaders.sentences_piboso_other()
print("PIBOSO sentences:", len(piboso))

pos = random.sample(civic, 2000) + random.sample(loaders.sentences_piboso_outcome(), 0)
neg = random.sample(abstracts, 2000) + random.sample(loaders.sentences_piboso_other(), 0)

X = pos + neg
y = ["pos"] * len(pos) + ["neg"] * len(neg)

# comment out for quick testing of existing model
model = dummy_pipeline.build_and_evaluate(X, y)  # , outpath="./dummy_clf.pickle")

print(dummy_pipeline.show_most_informative_features(model))

# ----------------------------------------------------------------

# unpickle classifier
# with open('./dummy_clf.pickle', 'rb') as f:
#     model = pickle.load(f)

preppy = preprocessors.BasicPreprocessor()

lab_civ = pd.DataFrame(data={"Label" : model.predict(civic),
                             "Text"  : civic,
                             "Tokens": preppy.transform(civic)},
                       columns=["Label", "Text", "Tokens"])

lab_civ.to_csv("./labelled_civic.csv")

lab_abs = pd.DataFrame(data={"Label" : model.predict(abstracts),
                             "Text"  : abstracts,
                             "Tokens": preppy.transform(abstracts)},
                       columns=["Label", "Text", "Tokens"])

lab_abs.to_csv("./labelled_abstracts.csv")

lab_oth = pd.DataFrame(data={"Label" : model.predict(loaders.sentences_piboso_other()),
                             "Text"  : loaders.sentences_piboso_other(),
                             "Tokens": preppy.transform(loaders.sentences_piboso_other())},
                       columns=["Label", "Text", "Tokens"])

lab_oth.to_csv("./labelled_other.csv")

lab_out = pd.DataFrame(data={"Label" : model.predict(loaders.sentences_piboso_outcome()),
                             "Text"  : loaders.sentences_piboso_outcome(),
                             "Tokens": preppy.transform(loaders.sentences_piboso_outcome())},
                       columns=["Label", "Text", "Tokens"])

lab_out.to_csv("./labelled_outcome.csv")

# save sentences with predicted labels to csv


# ----------------------------------------------------------------

yhat = model.predict([
    "Practical limitations to inducing neural regeneration are also addressed.",
    "The effects of glucocorticoids, lazeroids, gangliosides, opiate antagonists, calcium channel blockers, "
    "glutamate receptor antagonists, antioxidants, free radical scavengers, and other pharmacological agents in both "
    "animal models and human trials are summarized",
    "This is the worst movie I have ever seen!",
    "The movie was action packed and full of adventure!",
    "The pathogenesis of primary and secondary injury, as well as the theoretical bases of neurological recovery, "
    "are examined in detail.",
    "Bone, lean and fat mass were measured by dual-energy X-ray absorptiometry (DXA), and strength was measured using "
    "an isokinetic dynamometer.",
    "In girls, significant correlations were found between mass (lean, fat and body mass), strength and most bone "
    "characteristics (r = 0.15-0.93).",
    "At the proximal femur changes in bone mineral density (BMD) were moderately related to changes in body "
    "composition.",
    "Low to moderate correlations were observed between changes in bone and changes in body composition.",
    "Alendronate sodium (ALN) increases bone mineral density (BMD) in heterogeneous populations of postmenopausal "
    "women, but its effect is unknown in women with type 2 diabetes.",
    "Of these, 370 patients had information from their implanted cardiac resynchronization device for mortality risk "
    "stratification, and 288 patients had information for measured parameters (ie, HRV, night heart rate, and patient "
    "activity) and clinical event a "
    "This study demonstrates that SDAAM continuously measured from an implanted cardiac resynchronization device is "
    "lower in patients at high mortality and hospitalization risk.",
    "In conclusion, obesity and overweight are accompanied by unfavourable blood lipids patterns and in a "
    "considerable proportion of overweight or obese patients other risk factors for coronary heart disease, "
    "such as hypertension, smoking, diabetes or family h",
    "There were no statistically significant differences in rates or mean numbers of adverse events between "
    "paroxetine-treated patients and fluoxetine-treated patients.",
    "The subject's subsequent clinical course included a trial of gabapentin which produced a substantial reduction "
    "in frequency and average intensity of his episodic pain and which has been maintained for almost 2 years.",
    "Exogenous expression of the A842V mutation resulted in constitutive tyrosine phosphorylation of PDGFRA in the "
    "absence of ligand in 293T cells and cytokine-independent proliferation of the IL-3-dependent Ba/F3 cell line, "
    "both evidence that this is an activating mutation.",
    "JAK2 V617F is associated with myeloid malignanices (AML, MDS, CMML/atypical CML).",
    "cancer mutation summary no effect beneficial gene",
    "",
    "",
    civic[0], civic[10], civic[120], civic[210], civic[101], civic[130], civic[140], civic[190], civic[910],
    "",
    "",
    abstracts[0], abstracts[10], abstracts[120], abstracts[210], abstracts[101], abstracts[130], abstracts[20]
])
print(yhat)

print("civic: prediction", sum(lab_civ["Label"].values), "/", len(civic))

yhat2 = lab_abs
print("abstracts: prediction", sum(lab_abs["Label"].values), "/", len(yhat2))

yhat3 = model.predict(loaders.sentences_piboso_pop_bg_oth())
print("piboso pop bg oth: prediction", sum(yhat3), "/", len(yhat3))

yhat4 = model.predict(loaders.sentences_piboso_other())
print("piboso other: prediction", sum(yhat4), "/", len(yhat4))

yhat5 = model.predict(loaders.sentences_piboso_outcome())
print("piboso outcome: prediction", sum(yhat5), "/", len(yhat5))
