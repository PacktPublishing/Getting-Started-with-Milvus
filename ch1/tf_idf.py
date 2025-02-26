# Before running the script, you need to install the gensim python third-party library in advance.
# Execute: pip install gensim

from gensim import corpora, models, similarities

text_corpus = [
    "the cat in the hat",
    "the cat sat on the mat",
    "the dog sat on the log",
    "dogs and cats are great pets"
]

texts = [doc.lower().split() for doc in text_corpus]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

tfidf = models.TfidfModel(corpus)
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary))

query_document = 'cat hat'.split()
query_bow = dictionary.doc2bow(query_document)
sims = index[tfidf[query_bow]]
for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
    print(document_number, score)