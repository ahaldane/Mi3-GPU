#!/usr/bin/env python3
import numpy as np
import argparse, sys
from mi3gpu.utils.potts_common import getXij

def main():
    parser = argparse.ArgumentParser(
        description='Compute Xij scores')
    parser.add_argument('bimarg')
    parser.add_argument('J')
    parser.add_argument('--oXij', default='Xij', help="Output file for Xij")
    parser.add_argument('--oXijab', help="Output file for Xijab")

    args = parser.parse_args(sys.argv[1:])
    ff = np.load(args.bimarg)
    J = np.load(args.J)

    xij, xijab = getXij(J, ff)

    np.save(args.oXij, xij)
    if args.oXijab:
        np.save(args.oXijab, xijab)

if __name__ == '__main__':
    main()
