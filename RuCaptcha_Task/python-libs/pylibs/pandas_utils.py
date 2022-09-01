from pathlib import Path
from typing import Union

import pandas as pd


def read_dataframe(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    engine = kwargs.setdefault('engine', 'auto')
    return pd.read_parquet(path, engine=engine)


def write_dataframe(df: pd.DataFrame, path: Union[str, Path], **kwargs):
    engine = kwargs.setdefault('engine', 'auto')
    compression = kwargs.setdefault('compression', 'gzip')
    index = kwargs.setdefault('index', True)
    df.to_parquet(path, engine=engine, compression=compression, index=index)


DF_FILE_FORMAT = "%s.parquet.gzip"
