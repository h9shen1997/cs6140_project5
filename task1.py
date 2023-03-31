from collections import OrderedDict
from collections import deque

import numpy as np
import pandas as pd

market_sym = ['DJI', 'NASDAQ', 'NYSE', 'RUSSELL', 'S&P']
NUM_MARKET_SYM = 5


def read_data(file_prefix: str) -> None:
    """
    Reads the stock market technical analysis data and combine all 5 market index into one file.
    :param file_prefix: file prefix of all 5 market index.
    :return: None
    """
    filenames = []
    for suffix in market_sym:
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
        UP_LABEL = 0
        NO_CHANGE_LABEL = 1
        DOWN_LABEL = 2
        intelligent_fill(price_data)
        for k in range(len(price_data) - 1):
            if price_data[k] < price_data[k + 1]:
                price_dir.append(UP_LABEL)
            elif price_data[k] == price_data[k + 1]:
                price_dir.append(NO_CHANGE_LABEL)
            else:
                price_dir.append(DOWN_LABEL)
        all_data[f'{market_sym[i]}_price_dir'] = price_dir
        all_data[f'{market_sym[i]}_price'] = price_data

    all_data.to_csv('./CNNpred/day1prediction.csv', index=False)
    print(all_data.isna().any().any())


# def generate_price_dir(y_pred_list, y_test_list):
#     UP_LABEL = 1
#     NO_CHANGE_LABEL = 0
#     DOWN_LABEL = -1
#     y_pred_dir = []
#     y_test_dir = []
#     for i in range(len(y_pred_list) - 1):
#         if y_pred_list[i + 1] > y_pred_list[i]:
#             y_pred_dir.append(UP_LABEL)
#         elif y_pred_list[i + 1] < y_pred_list[i]:
#             y_pred_dir.append(DOWN_LABEL)
#         else:
#             y_pred_dir.append(NO_CHANGE_LABEL)
#         if y_test_list[i + 1] > y_test_list[i]:
#             y_test_dir.append(UP_LABEL)
#         elif y_test_list[i + 1] < y_test_list[i]:
#             y_test_dir.append(DOWN_LABEL)
#         else:
#             y_test_dir.append(NO_CHANGE_LABEL)
#     return y_pred_dir, y_test_dir


def intelligent_fill(l: list):
    n = len(l)
    queue = deque()
    for i in reversed(range(n)):
        if np.isnan(l[i]):
            cur_sum = 0.
            cur_len = len(queue) if len(queue) > 0 else 1
            for element in queue:
                cur_sum += element
            avg = cur_sum / cur_len
            l[i] = avg
        if len(queue) >= 5:
            queue.popleft()
        queue.append(l[i])


def train_test_split():
    SPLIT_INDEX = 1761
    df = pd.read_csv('./CNNpred/day1prediction.csv')
    train_df = df.iloc[:SPLIT_INDEX]
    test_df = df.iloc[SPLIT_INDEX:]
    train_df.to_csv('./CNNpred/day1prediction_train.csv', index=False)
    test_df.to_csv('./CNNpred/day1prediction_test.csv', index=False)


def main():
    pd.set_option('display.max_columns', 500)
    read_data('./CNNpred/Processed')
    train_test_split()


if __name__ == '__main__':
    main()
