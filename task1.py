"""
Filename: task1.py
Author: Haotian Shen, Qiaozhi Liu
"""
from collections import OrderedDict
from collections import deque

import numpy as np
import pandas as pd

from constants import NUM_MARKET_SYM, MARKET_SYM, SPLIT_INDEX, DISPLAY_MAX_COL, ESTIMATE_WINDOW_SIZE, UP_LABEL, \
    FLAT_LABEL, DOWN_LABEL


def read_data(file_prefix: str) -> None:
    """
    Reads the stock market technical analysis data and combine all 5 market index into one file.
    :param file_prefix: file prefix of all 5 market index.
    :return: None
    """
    filenames = []
    for suffix in MARKET_SYM:
        filename = f'{file_prefix}_{suffix}.csv'
        filenames.append(filename)

    df_list = []

    for i in range(NUM_MARKET_SYM):
        df_list.append(pd.read_csv(filenames[i]))
        df_list[i].drop(columns='Name', axis=1, inplace=True)

    for i in range(NUM_MARKET_SYM):
        print(df_list[i].shape)

    feature_columns = []
    for i in range(NUM_MARKET_SYM):
        feature_columns = feature_columns + df_list[i].columns.tolist()
    feature_columns = list(OrderedDict.fromkeys(feature_columns))

    all_data = pd.DataFrame()
    for feature in feature_columns:
        if feature == 'Close':
            continue
        cur_col = None
        for i in range(NUM_MARKET_SYM):
            cur_list = df_list[i]
            if feature in cur_list.columns:
                if cur_col is None:
                    cur_col = cur_list[feature].tolist()
                else:
                    cur_col_val = cur_list[feature].tolist()
                    for j in range(len(cur_col)):
                        if cur_col[j] is None:
                            cur_col[j] = cur_col_val[j]
        if feature != 'Date':
            intelligent_fill(cur_col)
        all_data[feature] = cur_col

    for i in range(5):
        price_data = df_list[i]['Close'].tolist()
        price_dir = [1]
        intelligent_fill(price_data)
        for k in range(len(price_data) - 1):
            if price_data[k] < price_data[k + 1]:
                price_dir.append(UP_LABEL)
            elif price_data[k] == price_data[k + 1]:
                price_dir.append(FLAT_LABEL)
            else:
                price_dir.append(DOWN_LABEL)
        all_data[f'{MARKET_SYM[i]}_price_dir'] = price_dir
        all_data[f'{MARKET_SYM[i]}_price'] = price_data

    all_data.to_csv('./CNNpred/day1prediction.csv', index=False)
    print(all_data.isna().any().any())


def intelligent_fill(list_with_na: list):
    """
    Replaces NaN values in a list using a simple heuristic.
    :param list_with_na: A list containing NaN values to be replaced
    :returns: None
    This function fills in missing values in a list with NaN values. It works by using a sliding window approach to calculate
    the average of the non-NaN values and uses that as an estimate for the NaN value. Specifically, it takes the previous
    ESTIMATE_WINDOW_SIZE non-NaN values and calculates their average. If there are fewer than ESTIMATE_WINDOW_SIZE non-NaN
    values available, it uses the average of the non-NaN values available. This average is used as an estimate for the NaN value.
    The function replaces NaN values in the original list with the estimated values.
    """
    n = len(list_with_na)
    queue = deque()
    for i in reversed(range(n)):
        if np.isnan(list_with_na[i]):
            cur_sum = 0.
            cur_len = len(queue) if len(queue) > 0 else 1
            for element in queue:
                cur_sum += element
            avg = cur_sum / cur_len
            list_with_na[i] = avg
        if len(queue) >= ESTIMATE_WINDOW_SIZE:
            queue.popleft()
        queue.append(list_with_na[i])


def train_test_split():
    """
    Reads in a CSV file located at './CNNpred/day1prediction.csv' and splits it into training and testing data using SPLIT_INDEX.
    The training data is saved to './CNNpred/day1prediction_train.csv', and the testing data is saved to './CNNpred/day1prediction_test.csv'.
    :return: None
    """
    df = pd.read_csv('./CNNpred/day1prediction.csv')
    train_df = df.iloc[:SPLIT_INDEX]
    test_df = df.iloc[SPLIT_INDEX:]
    train_df.to_csv('./CNNpred/day1prediction_train.csv', index=False)
    test_df.to_csv('./CNNpred/day1prediction_test.csv', index=False)


def main():
    """
    Main function.
    :return: None
    """
    pd.set_option('display.max_columns', DISPLAY_MAX_COL)
    read_data('./CNNpred/Processed')
    train_test_split()


if __name__ == '__main__':
    main()
