from semisuper import loaders

civic, abstracts = loaders.sentences_civic_abstracts()

for c in zip(civic, range(len(civic))):
    s = "./sentences/c" + str(c[1]) + ".txt"
    with open(s, 'w') as f:
        f.write(c[0])

for a in zip(abstracts, range(len(abstracts))):
    s = "./sentences/a" + str(a[1]) + ".txt"
    with open(s, 'w') as f:
        f.write(a[0])
