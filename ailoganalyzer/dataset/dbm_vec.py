import shelve
from tqdm import tqdm


def install_vectors(fname_in, fname_out):
    d = shelve.open(fname_out, "c")
    
    with open(fname_in, "r") as f:
        num_lines = int(f.readline().split()[0])
    f=True
    
    with open(fname_in, "r") as f:
        for line in tqdm(f, total=num_lines):
            if not f:
                tokens = line.rstrip().split(' ')
                d[tokens[0]] = [float(e) for e in tokens[1:]]
            else:
                f=False

    d.close()
