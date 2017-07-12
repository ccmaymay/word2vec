#!/usr/bin/env python3


import io
import os
from contextlib import contextmanager

import pandas as pd

import matplotlib
matplotlib.use('PDF')  # noqa
import matplotlib.pyplot as plt


def mkdirp(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def parse_tab(input_path):
    return pd.read_table(input_path)


@contextmanager
def plot(output_path, title=None, xlabel=None, ylabel=None):
    yield
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.savefig(output_path)
    plt.clf()


def plot_runtime_mean_per_configuration(df, output_path):
    with plot(
            output_path,
            title='mean runtime of each configuration',
            xlabel='configuration',
            ylabel='mean runtime',
            ):
        df.groupby('output_path')['wallclock_seconds'].mean().plot.bar()


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='ingest word2vec runtime data, plot',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='path to input tab file')
    parser.add_argument(
        'output_dir', help='path to directory where output will be written')
    args = parser.parse_args()

    df = parse_tab(args.input_path)

    mkdirp(args.output_dir)

    plot_runtime_mean_per_configuration(
        df,
        os.path.join(args.output_dir, 'runtime_mean_per_configuration.pdf'))


if __name__ == '__main__':
    main()
