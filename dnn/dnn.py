import numpy as np
import pandas as pd
import keras
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler

def df_to_ds(dataframe, input_features, label=None):
    input_data = []
    for feature in input_features:
        input_data.append(dataframe[feature].to_numpy())
    input_data = np.array(input_data).T
    input_data = np.reshape(input_data, (input_data.shape[0], input_data.shape[1], 1))
    if label:
        label = dataframe[label[0]].to_numpy()
    return input_data, label

def train_val_split(dataframe, train_ratio):
    train_df = dataframe.iloc[:int(dataframe.shape[0]*train_ratio),:]
    val_df = dataframe.iloc[int(dataframe.shape[0]*train_ratio):,:]
    return train_df, val_df

def build_RNN(input_shape_):
    #Initializing the RNN
    model = Sequential()
    #Making a robust stacked LSTM layer
    #Adding the first LSTM layer and some Dropout Regularization
    model.add(LSTM(units = 50, return_sequences = True, input_shape = input_shape_))
    model.add(Dropout(0.2))
    #Adding the second LSTM layer and some Dropout Regularization
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    #Adding the third LSTM layer and some Dropout Regularization
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    #Adding the FOURTH LSTM layer and some Dropout Regularization
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    #Adding the fifth LSTM layer and some Dropout Regularization
    model.add(LSTM(units = 50, return_sequences = False))
    model.add(Dropout(0.2))
    #Adding the output layer
    model.add(Dense(units = 1))

    return model

if __name__ == "__main__":
    
    input_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    label = ['Target']

    df = pd.read_csv("../train_files/stock_prices.csv")
    #df = df.iloc[:5,:]
    test_df = pd.read_csv("../supplemental_files/stock_prices.csv")
    
    df.dropna(subset=input_features + label, axis=0, inplace=True)
    test_df.dropna(subset=input_features, axis=0, inplace=True)

    train_df,val_df = train_val_split(df, 0.8)

    input_train, label_train = df_to_ds(train_df, input_features, label=label)
    input_val, label_val = df_to_ds(val_df, input_features, label=label)
    X_test, label = df_to_ds(test_df, input_features)

    model = build_RNN((input_train.shape[1], 1))

    #Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

    model.summary()
    plot_model(model, show_shapes=True, rankdir="LR")

    #Fitting the RNN to the Training set
    model.fit(input_train, label_train, epochs = 3)  
    model.save('stock_prediction_rnn.h5')

    stock_prediction_rnn = model.predict(X_test)
