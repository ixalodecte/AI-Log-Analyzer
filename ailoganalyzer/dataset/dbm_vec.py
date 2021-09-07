import dbm
import io
from tqdm import tqdm
import pathlib
import sys


def install_vectors(fname):
    vec_file = str(pathlib.Path(__file__).parent.resolve() / "vec")
    d = dbm.open(vec_file, 'c')
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        d[tokens[0]] = " ".join(tokens[1:])
    d.close()
