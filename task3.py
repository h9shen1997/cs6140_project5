"""
Filename: task3.py
Author: Haotian Shen, Qiaozhi Liu
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import INPUT_DIM, OUTPUT_DIM, SEED_NUMBER, MARKET_SYM, DEFAULT_TIME_STEPS
from task2 import train, test, preprocess_data, read_file


class StockPredictorCNN(nn.Module):
    """
    Implementation of a Convolutional Neural Network for stock price prediction.

    :param input_channels: Number of input channels.
    :type input_channels: int
    :param conv1_out_channels: Number of output channels for the first convolutional layer.
    :type conv1_out_channels: int
    :param num_classes: Number of classes for the classification task. Default is 3.
    :type num_classes: int
    :param time_steps: Number of time steps in the input data. Default is DEFAULT_TIME_STEPS.
    :type time_steps: int

    :ivar conv1: First convolutional layer.
    :vartype conv1: nn.Conv1d
    :ivar conv2: Second convolutional layer.
    :vartype conv2: nn.Conv1d
    :ivar fc1: First fully connected layer.
    :vartype fc1: nn.Linear
    :ivar fc2: Second fully connected layer.
    :vartype fc2: nn.Linear

    :return: Output tensor of shape (batch_size, num_classes).
    :rtype: torch.Tensor
    """

    def __init__(self, input_channels, conv1_out_channels, num_classes=3, time_steps=DEFAULT_TIME_STEPS):
        super(StockPredictorCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, conv1_out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv1_out_channels, conv1_out_channels // 2, kernel_size=3, padding=1)
        fc1_input_channels = time_steps // 2 * conv1_out_channels // 2
        self.fc1 = nn.Linear(fc1_input_channels, 60)
        self.fc2 = nn.Linear(60, num_classes)

    def forward(self, x):
        """
        Forward pass of the StockPredictorCNN.

        :param x: Input tensor of shape (batch_size, input_channels, time_steps).
        :type x: torch.Tensor

        :return: Output tensor of shape (batch_size, num_classes).
        :rtype: torch.Tensor
        """
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_dataset(X, y, window_size=DEFAULT_TIME_STEPS):
    """
    This function creates a dataset for time series analysis by converting the input data and target labels into arrays of windows.

    :param X: pandas dataframe, the input data for time series analysis
    :param y: pandas series, the target labels for time series analysis
    :param window_size: int, the size of the window to be used for creating arrays of input data and target labels. Defaults to DEFAULT_TIME_STEPS.

    :return: tuple of numpy arrays, where the first element is an array of input data and the second element is an array of target labels.

    The function loops through the input data and target labels and creates arrays of windows of length 'window_size' for each data point. These arrays of windows are stored in 'Xs' and 'ys' respectively. The function returns 'Xs' and 'ys' as numpy arrays.
    """
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X.iloc[i:(i + window_size)].values)
        ys.append(y.iloc[i + window_size])
    return np.array(Xs), np.array(ys)


def main():
    """
    Main function.
    :return: None
    """
    model = StockPredictorCNN(INPUT_DIM, 60, OUTPUT_DIM)

    train_df = read_file('./CNNpred/day1prediction_train.csv')
    test_df = read_file('./CNNpred/day1prediction_test.csv')
    # drop the first that contains the year 2009 data.
    train_df = train_df.drop(0)

    market_acc = []
    print('Using cnn model...')
    for market in MARKET_SYM:
        print(f'Predicting {market} price...')
        X_train, y_train = preprocess_data(train_df, market)
        X_test, y_test = preprocess_data(test_df, market)

        # get a quick peek of the data shape
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        X_train_3D, y_train_3D = create_dataset(X_train, y_train, DEFAULT_TIME_STEPS)
        X_test_3D, y_test_3D = create_dataset(X_test, y_test, DEFAULT_TIME_STEPS)

        # prepare tensor from dataframe
        X_train_tensor = torch.tensor(X_train_3D, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_3D, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_3D, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test_3D, dtype=torch.long)

        # use cross entropy loss for multi-class classification
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # train the model on the training set
        train(X_train_tensor, y_train_tensor, model, criterion, optimizer, epochs=10)

        # Evaluate the model on the test set
        acc = test(X_test_tensor, y_test_tensor, model)

        print(f'Price direction prediction accuracy: {acc}')
        market_acc.append(acc)

    print(f'Average price direction prediction accuracy: {sum(market_acc) / len(market_acc)}')
    print(pd.DataFrame({'market': MARKET_SYM, 'accuracy': market_acc}).to_string(index=False))


if __name__ == '__main__':
    torch.manual_seed(SEED_NUMBER)
    main()
