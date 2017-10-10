import preprocessor
import dummy_classifier
import pickle
import loader
import random

civic, abstracts = loader.sentences_civic_abstracts()

print("civic sentences:", len(civic))
print("abstract sentences:", len(abstracts))

piboso = loader.sentences_piboso_pop_bg_oth()
print("PIBOSO sentences:", len(piboso))

pos = random.sample(civic, 2000)
neg = random.sample(piboso, 2000)

X = pos + neg
y = ["yay"] * len(pos) + ["neg"] * len(neg)

# comment out for quick testing of existing model
model = dummy_classifier.build_and_evaluate(X, y, outpath="./dummy_clf.pickle")

print(dummy_classifier.show_most_informative_features(model))

with open('./dummy_clf.pickle', 'rb') as f:
    model = pickle.load(f)

yhat = model.predict([
    "Practical limitations to inducing neural regeneration are also addressed.",
    "The effects of glucocorticoids, lazeroids, gangliosides, opiate antagonists, calcium channel blockers, glutamate receptor antagonists, antioxidants, free radical scavengers, and other pharmacological agents in both animal models and human trials are summarized",
    "This is the worst movie I have ever seen!",
    "The movie was action packed and full of adventure!",
    "The pathogenesis of primary and secondary injury, as well as the theoretical bases of neurological recovery, are examined in detail.",
    "Bone, lean and fat mass were measured by dual-energy X-ray absorptiometry (DXA), and strength was measured using an isokinetic dynamometer.",
    "In girls, significant correlations were found between mass (lean, fat and body mass), strength and most bone characteristics (r = 0.15-0.93).",
    "At the proximal femur changes in bone mineral density (BMD) were moderately related to changes in body composition.",
    "Low to moderate correlations were observed between changes in bone and changes in body composition.",
    "Alendronate sodium (ALN) increases bone mineral density (BMD) in heterogeneous populations of postmenopausal women, but its effect is unknown in women with type 2 diabetes.",
    "Of these, 370 patients had information from their implanted cardiac resynchronization device for mortality risk stratification, and 288 patients had information for measured parameters (ie, HRV, night heart rate, and patient activity) and clinical event a"
    "This study demonstrates that SDAAM continuously measured from an implanted cardiac resynchronization device is lower in patients at high mortality and hospitalization risk.",
    "In conclusion, obesity and overweight are accompanied by unfavourable blood lipids patterns and in a considerable proportion of overweight or obese patients other risk factors for coronary heart disease, such as hypertension, smoking, diabetes or family h",
    "There were no statistically significant differences in rates or mean numbers of adverse events between paroxetine-treated patients and fluoxetine-treated patients.",
    "The subject's subsequent clinical course included a trial of gabapentin which produced a substantial reduction in frequency and average intensity of his episodic pain and which has been maintained for almost 2 years.",
    "Exogenous expression of the A842V mutation resulted in constitutive tyrosine phosphorylation of PDGFRA in the absence of ligand in 293T cells and cytokine-independent proliferation of the IL-3-dependent Ba/F3 cell line, both evidence that this is an activating mutation.",
    "JAK2 V617F is associated with myeloid malignanices (AML, MDS, CMML/atypical CML).",
    "cancer mutation summary no effect beneficial gene",
    "",
    "",
    civic[0], civic[10], civic[120], civic[210], civic[101], civic[130], civic[140], civic[190], civic[910],
    "",
    "",
    abstracts[0], abstracts[10], abstracts[120], abstracts[210], abstracts[101], abstracts[130], abstracts[0]
])

print(yhat)

yhat1 = model.predict(civic)
print(sum(yhat1))

yhat2 = model.predict(abstracts)
print(sum(yhat2))

yhat3 = model.predict(piboso)
print(sum(yhat3))
