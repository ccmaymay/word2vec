#!/usr/bin/env python3


import io


def main():
    import sys
    print('\t'.join((
        'host',
        'atlas_run_1', 'atlas_run_2', 'atlas_run_3',
        'openblas_run_1', 'openblas_run_2', 'openblas_run_3'
    )))
    paths = sys.argv[1:]
    for path in paths:
        host_data = []
        with io.open(path) as f:
            for line in f:
                line = line.strip()
                pieces = line.split('\t')
                if not host_data:
                    host_data.append(line)
                elif len(pieces) == 11 and \
                        not line.startswith('wallclock_seconds'):
                    host_data.append(pieces[0])
        print('\t'.join(host_data))


if __name__ == '__main__':
    main()
