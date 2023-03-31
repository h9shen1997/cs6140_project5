import pandas as pd
import torch
import torch.nn as nn

from task1 import market_sym


def read_file(filename: str):
    df = pd.read_csv(filename)
    return df


class FFNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        # x = x.view(x.size(0), 1, x.size(1))
        x, hn = self.rnn(x, h0)
        x = self.fc(x[:, -1, :])
        return x


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # x = x.view(x.size(0), 1, x.size(1))
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = self.fc(x[:, -1, :])
        return x


def preprocess_data(df, market_sym):
    X = df.drop('Date', axis=1)
    X_col_to_drop = X.filter(regex='(price|price_dir)$', axis=1).columns
    X = X.drop(X_col_to_drop, axis=1)

    y = df[f'{market_sym}_price_dir']
    return X, y


def train(X_tensor, y_tensor, model, loss_fn, optimizer, epochs=10):
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = loss_fn(y_pred, y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, train loss: {loss.item():.4f}")


def test(X_tensor, y_tensor, model, loss_fn):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor)
        y_pred_class = torch.argmax(y_pred, dim=1)
        acc = (y_pred_class == y_tensor).sum().item() / len(y_tensor)
        print(f'Test acc: {acc:.4f}')
        return acc


def main():
    INPUT_DIM = 83
    OUTPUT_DIM = 3
    model_dict = {'ffnn': FFNN(INPUT_DIM, OUTPUT_DIM), 'rnn': RNN(INPUT_DIM, 64, OUTPUT_DIM),
                  'lstm': LSTM(INPUT_DIM, 64, 2, OUTPUT_DIM)}

    train_df = read_file('./CNNpred/day1prediction_train.csv')
    test_df = read_file('./CNNpred/day1prediction_test.csv')
    # drop the first that contains the year 2009 data.
    train_df = train_df.drop(0)

    for model_name in ['ffnn', 'rnn', 'lstm']:
        market_acc = []
        print(f'Using {model_name} model...')
        model = model_dict[model_name]
        for market in market_sym:
            print(f'Predicting {market} price...')
            X_train, y_train = preprocess_data(train_df, market)
            X_test, y_test = preprocess_data(test_df, market)

            # get a quick peek of the data shape
            print(X_train.shape)
            print(X_test.shape)
            print(y_train.shape)
            print(y_test.shape)

            # prepare tensor from dataframe
            X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
            X_train_tensor = X_train_tensor.reshape(X_train_tensor.shape[0], 1, X_train_tensor.shape[1])
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
            X_test_tensor = X_test_tensor.reshape(X_test_tensor.shape[0], 1, X_test_tensor.shape[1])
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

            # use cross entropy loss for multi-class classification
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # train the model on the training set
            train(X_train_tensor, y_train_tensor, model, criterion, optimizer)

            # Evaluate the model on the test set
            acc = test(X_test_tensor, y_test_tensor, model, criterion)

            print(f'Price direction prediction accuracy: {acc}')
            market_acc.append(acc)

        print(f'Average price direction prediction accuracy: {sum(market_acc) / len(market_acc)}')
        print(pd.DataFrame({'market': market_sym, 'accuracy': market_acc}).to_string(index=False))


if __name__ == '__main__':
    seed_number = 42
    torch.manual_seed(seed_number)
    main()
