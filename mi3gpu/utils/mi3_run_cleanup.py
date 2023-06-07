#!/usr/bin/env python
import glob
from pathlib import Path
import argparse

files = ['J.npy', 'jstep', 'info.txt', 'nsteps', 'seqs']
extra_files = ['bicounts','bimarg.npy','predictedBimarg.npy', 
               'energies.npy', 'perturbedJ.npy']
def iscomplete(rundir):
    # check that all important files are there
    for f in files:
        if not (rundir / f).exists():
            return False
    return True
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=Path)
    parser.add_argument('--rmseqs', action='store_true')
    parser.add_argument('--rmJ', action='store_true')
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()

    # clean up all but the last complete folder of runs
    rundirs = sorted(args.dir.glob('run_*'))
    inc = []
    while not iscomplete(rundirs[-1]):
        inc.append(rundirs[-1])
        rundirs.pop()

    if args.rmseqs:
        extra_files.append('seqs')
    if args.rmJ:
        extra_files.append('J.npy')

    for d in rundirs[:-1]:
        for f in extra_files:
            r = d / f
            if not r.exists():
                print("skipped ", r, ", not present")
                continue
            print("removing ", r)
            if not args.dry:
                r.unlink()

    print("")
    for i in inc:
        print(f"Warning: run {i} appears to be incomplete, skipped")
    print("Skipped last run", rundirs[-1])

if __name__ == '__main__':
    main()
