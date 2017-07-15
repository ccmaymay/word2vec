#!/usr/bin/env python


from __future__ import unicode_literals
import io

from numpy.random import poisson


BUFFER_SIZE = 1024


def iter_words(f):
    text = f.read(BUFFER_SIZE)
    word = ''
    while text:
        for c in text:
            if c == '\r':
                continue
            elif c == ' ' or c == '\t' or c == '\n':
                if word:
                    yield word
                    word = ''
            else:
                word += c
        text = f.read(BUFFER_SIZE)


def draw_sentence_length(poisson_lambda):
    sentence_length = 0
    while sentence_length == 0:
        sentence_length = poisson(poisson_lambda)
    return sentence_length


def split_into_sentences(f, poisson_lambda):
    sentence = []
    sentence_length = draw_sentence_length(poisson_lambda)

    for word in iter_words(f):
        sentence.append(word)
        if len(sentence) == sentence_length:
            yield sentence
            sentence = []
            sentence_length = draw_sentence_length(poisson_lambda)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    parser.add_argument('--poisson-lambda', type=float, default=15.)
    args = parser.parse_args()

    with io.open(args.input_path) as in_f:
        with io.open(args.output_path, 'w') as out_f:
            for sentence in split_into_sentences(in_f, args.poisson_lambda):
                out_f.write(' '.join(sentence) + '\n')


if __name__ == '__main__':
    main()
