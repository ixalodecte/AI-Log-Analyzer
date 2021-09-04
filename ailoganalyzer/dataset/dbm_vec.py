import dbm
import io
from tqdm import tqdm


def install_vectors(fname):
    d = dbm.open('vec', 'c')
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        d[tokens[0]] = " ".join(tokens[1:])
    d.close()
