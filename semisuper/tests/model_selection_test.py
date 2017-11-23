from semisuper import loaders, pu_two_step, pu_biased_svm, basic_pipeline, pu_model_selection
from semisuper.helpers import num_rows, unsparsify
import random
from sklearn.model_selection import train_test_split

civic, abstracts = loaders.sentences_civic_abstracts()
hocpos, hocneg = loaders.sentences_HoC()
piboso_other = loaders.sentences_piboso_other()
piboso_outcome = loaders.sentences_piboso_outcome()

print("CIViC sentences:", len(civic))
print("Abstract sentences:", len(abstracts))
print("HoC positive sentences:", len(hocpos))
print("HoC negative sentences:", len(hocneg))
print("PIBOSO outcome sentences:", len(piboso_outcome))
print("PIBOSO other sentences:", len(piboso_other))

hocpos_train, hocpos_test = train_test_split(hocpos, test_size=0.3)
hocneg_train, hocneg_test = train_test_split(hocneg, test_size=0.3)
civic_train, civic_test = train_test_split(civic, test_size=0.3)
abstracts_train, abstracts_test = train_test_split(abstracts, test_size=0.01)

P = random.sample(hocpos_train + civic_train, 2000)
U = random.sample(hocneg_train + abstracts_train, 2000)

half_test_size=200
X_test = random.sample(hocpos_test, half_test_size) + random.sample(hocneg_test, half_test_size)
y_test = [1] * half_test_size + [0] * half_test_size

pu_model_selection.getBestModel(P, U, X_test, y_test)
