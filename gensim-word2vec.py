#!/usr/bin/env python

from gensim.models.word2vec import Word2Vec, LineSentence

if __name__ == '__main__':
    sentences = LineSentence('text8', max_sentence_length=1000)
    model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=1,
                     alpha=0.025, sample=0.001, min_alpha=0.025 * 0.0001,
                     negative=5, iter=1, sg=1)
    model.save('/dev/null')
