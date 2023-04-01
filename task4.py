import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from constants import SEED_NUMBER, MARKET_SYM
from task2 import train, test, read_file
from task3 import StockPredictorCNN, create_dataset


def reprocess_fine_grained(df: pd.DataFrame):
    BIG_DOWN_LABEL = 0
    SMALL_DOWN_LABEL = 1
    FLAT_LABEL = 2
    SMALL_UP_LABEL = 3
    BIG_UP_LABEL = 4
    all_market_price_dir = []
    for market in MARKET_SYM:
        market_price_dir = [2]
        market_price = df[f'{market}_price'].tolist()
        for k in range(len(market_price) - 1):
            percent_change = (market_price[k + 1] - market_price[k]) / market_price[k]
            if percent_change < -0.015:
                market_price_dir.append(BIG_DOWN_LABEL)
            elif -0.015 <= percent_change < 0:
                market_price_dir.append(SMALL_DOWN_LABEL)
            elif percent_change == 0:
                market_price_dir.append(FLAT_LABEL)
            elif 0 < percent_change <= 0.015:
                market_price_dir.append(SMALL_UP_LABEL)
            elif percent_change > 0.015:
                market_price_dir.append(BIG_UP_LABEL)
        all_market_price_dir.append(market_price_dir)
    for i in range(len(MARKET_SYM)):
        df[f'{MARKET_SYM[i]}_precise_price_dir'] = all_market_price_dir[i]


def preprocess_data(df, market_sym):
    X = df.drop('Date', axis=1)
    X_col_to_drop = X.filter(regex='(price|price_dir|precise_price_dir)', axis=1).columns
    X = X.drop(X_col_to_drop, axis=1)

    y = df[f'{market_sym}_precise_price_dir']
    return X, y


def main():
    # fine-grained classification using 5 classes
    NEW_NUM_CLASSES = 5
    # use more history days
    MORE_TIME_STEPS = 30

    INPUT_DIM = 37
    # INPUT_DIM = 83

    # use a more complex model by increasing the first convolution layer's output channels
    model = StockPredictorCNN(INPUT_DIM, 160, NEW_NUM_CLASSES, MORE_TIME_STEPS)

    train_df = read_file('./CNNpred/day1prediction_train.csv')
    test_df = read_file('./CNNpred/day1prediction_test.csv')
    test_df.loc[-1] = train_df.iloc[-1]
    test_df.index = test_df.index + 1
    test_df = test_df.sort_index()
    reprocess_fine_grained(train_df)
    reprocess_fine_grained(test_df)

    # drop the first that contains the year 2009 data.
    train_df = train_df.drop(0)
    test_df = test_df.drop(0)

    scaler = StandardScaler()
    pca = PCA(n_components=0.95)

    market_acc = []
    print('Using cnn model...')
    for market in MARKET_SYM:
        print(f'Predicting {market} price...')
        X_train, y_train = preprocess_data(train_df, market)
        X_test, y_test = preprocess_data(test_df, market)

        # normalize the data
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # use PCA to reduce the dimensionality of the data
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)

        # get a quick peek of the data shape
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        X_train_3D, y_train_3D = create_dataset(X_train, y_train, MORE_TIME_STEPS)
        X_test_3D, y_test_3D = create_dataset(X_test, y_test, MORE_TIME_STEPS)

        # prepare tensor from dataframe
        X_train_tensor = torch.tensor(X_train_3D, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_3D, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_3D, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test_3D, dtype=torch.long)

        # use cross entropy loss for multi-class classification
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # train the model on the training set
        train(X_train_tensor, y_train_tensor, model, criterion, optimizer, epochs=50)

        # Evaluate the model on the test set
        acc = test(X_test_tensor, y_test_tensor, model)

        print(f'Price direction prediction accuracy: {acc}')
        market_acc.append(acc)

    print(f'Average price direction prediction accuracy: {sum(market_acc) / len(market_acc)}')
    print(pd.DataFrame({'market': MARKET_SYM, 'accuracy': market_acc}).to_string(index=False))


if __name__ == '__main__':
    torch.manual_seed(SEED_NUMBER)
    main()
