# this script contains functions used for general stuff, such as reading a (.csv) file into a dataframe
import numpy as np
import pandas as pd


def read_csv_to_df

def read_csv_to_np(path: str):
    df = pd.read_csv(path)
    raw_data = df.values
    return raw_data

