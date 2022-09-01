import pandas as pd
import numpy as np


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def prepare_data_array(df: pd.DataFrame):
    chunk_size = 60
    df = df.drop(columns='date')
    x_data = []
    for chunk in chunks(df, chunk_size):
        feature_vector = []
        for row in chunk.itertuples():
            feature_vector.extend(row[1:])
        x_data.append(np.asarray(feature_vector))
    return np.asarray(x_data)
