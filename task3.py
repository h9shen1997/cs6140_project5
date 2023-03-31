import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from task1 import market_sym
from task2 import train, test, preprocess_data, read_file

time_steps = 5


class StockPredictorCNN(nn.Module):
    def __init__(self, input_channels, num_classes=3):
        super(StockPredictorCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_dataset(X, y, time_steps=5):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


def main():
    INPUT_DIM = 83
    OUTPUT_DIM = 3
    model = StockPredictorCNN(INPUT_DIM, OUTPUT_DIM)

    train_df = read_file('./CNNpred/day1prediction_train.csv')
    test_df = read_file('./CNNpred/day1prediction_test.csv')
    # drop the first that contains the year 2009 data.
    train_df = train_df.drop(0)

    market_acc = []
    print('Using cnn model...')
    for market in market_sym:
        print(f'Predicting {market} price...')
        X_train, y_train = preprocess_data(train_df, market)
        X_test, y_test = preprocess_data(test_df, market)

        # get a quick peek of the data shape
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        X_train_3D, y_train_3D = create_dataset(X_train, y_train, time_steps)
        X_test_3D, y_test_3D = create_dataset(X_test, y_test, time_steps)

        # prepare tensor from dataframe
        X_train_tensor = torch.tensor(X_train_3D, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_3D, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_3D, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test_3D, dtype=torch.long)

        # use cross entropy loss for multi-class classification
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # train the model on the training set
        train(X_train_tensor, y_train_tensor, model, criterion, optimizer, epochs=30)

        # Evaluate the model on the test set
        acc = test(X_test_tensor, y_test_tensor, model)

        print(f'Price direction prediction accuracy: {acc}')
        market_acc.append(acc)

    print(f'Average price direction prediction accuracy: {sum(market_acc) / len(market_acc)}')
    print(pd.DataFrame({'market': market_sym, 'accuracy': market_acc}).to_string(index=False))


if __name__ == '__main__':
    seed_numer = 42
    torch.manual_seed(seed_numer)
    main()
