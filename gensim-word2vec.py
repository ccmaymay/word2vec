#!/usr/bin/env python


from __future__ import unicode_literals
import logging

from gensim.models.word2vec import Word2Vec, LineSentence


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('operation', choices=('build-vocab', 'train-model'))
    parser.add_argument('data_path')
    parser.add_argument('vocab_path')
    parser.add_argument('model_path', nargs='?')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
    )

    if args.operation == 'build-vocab':
        sentences = LineSentence(args.data_path, max_sentence_length=1000)
        print('initializing vocab')
        model = Word2Vec(size=100, window=5, min_count=5, workers=1,
                         alpha=0.025, sample=0.001, min_alpha=0.025 * 0.0001,
                         negative=5, iter=1, sg=1, hs=0)
        print('building vocab')
        model.build_vocab(sentences)
        print('saving vocab')
        model.save(args.vocab_path)

    elif args.operation == 'train-model':
        sentences = LineSentence(args.data_path, max_sentence_length=1000)
        print('loading vocab')
        model = Word2Vec.load(args.vocab_path)
        print('training model')
        model.train(sentences, total_examples=model.corpus_count,
                    epochs=model.iter)
        if args.model_path:
            print('saving model')
            model.save(args.model_path)

    else:
        raise ValueError('unknown operation {}'.format(args.operation))


if __name__ == '__main__':
    main()
