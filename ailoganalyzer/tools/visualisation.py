from pylab import show
import pandas as pd


def structured_to_time_serie(seq_filename):

    data = pd.read_csv(seq_filename)
    data["sequence"] = data["sequence"].transform(lambda x: len(x.strip('][').split(', ')))
    data["time"] = pd.to_datetime(data["time"], format="%Y-%m-%d %H:%M:%S.%f")
    return data


def visualize_time_serie(time_serie):
    time_serie.plot("time", "sequence")
    show()
