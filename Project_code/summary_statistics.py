# this script will be used to calculate the summary statistics of attributes in a dataset.
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict


@dataclass()
class SumStats:
    attr_name: str
    col_idx: int
    mean: float = 0.0
    std: float = 0.0
    obs: float = 0.0
    quantiles: list = field(default_factory=list)

    def __repr__(self):
        """
        Function for representing a class instance as a string. Called when using print() on an instance of the class.
        :return:
        """
        prec = 3 # decimal precision
        stats = f"mean = {round(self.mean,prec)}. std = {round(self.std,prec)}. obs = {round(self.obs,prec)}"
        # using list comprehension to remove scientific notation from representation
        stats += f", quantiles (q0,q0.5,q1) = {[round(q,prec) for q in self.quantiles]}"
        return f"{self.attr_name}: {stats}"

def calculate_summary_stats(df: pd.DataFrame, attribute_names: [str]):
    """
    :param df: a pandas dataframe
    :param attribute_names: a string list
    :return: dictionary of summary statistics for each attribute in attribute_names.
    Summary statistics only regards numerical attributes, & includes:
    Median, Mean, Min, Max, Standard Deviation, number of observations.
    """
    summ_stat_names = ("median", "mean", "min", "max", "st.d", "obs")
    # for attribute in attribute names:
    # calculate the summary statistic of that row
    df_columns = list(df.columns)
    summ_stats = [SumStats(attr_name, col_idx=df_columns.index(attr_name)) for \
                  attr_name in attribute_names]
    data = df.values

    # helper function for checking if value is numemrical:
    def is_numerical(value):
        try:
            float(value)
            return True
        except ValueError as e:
            return False

    # iterate through the columns and only consider the rows with numerical data
    for attr in summ_stats:
        col_idx = attr.col_idx
        col_data = data[:, col_idx]
        col_data = np.asarray([float(value) for value in col_data if is_numerical(value)])
        # assign the summary statistics:
        attr.obs = len(col_data)
        attr.mean = col_data.mean()
        quantile_percentages = [0, 0.5, 1.0]
        attr.quantiles = np.quantile(col_data, q=quantile_percentages)
        attr.std = col_data.std()

    return summ_stats
