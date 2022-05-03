import numpy as np
import pandas as pd
import keras
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler

def df_to_ds(dataframe, input_features, label):
    input_data = []
    for feature in input_features:
        input_data.append(dataframe[feature].to_numpy())
    input_data = np.array(input_data).T
    input_data = np.reshape(input_data, (input_data.shape[0], input_data.shape[1], 1))
    label = dataframe[label].to_numpy()
    # label = np.reshape(label, (label.shape[0], 1))
    return input_data, label

if __name__ == "__main__":
    df = pd.read_csv("../train_files/stock_prices.csv")
    df = df.sort_values("SecuritiesCode")
    securities_count = df["SecuritiesCode"].value_counts()
    securities = securities_count.to_dict()
    
    list_df = []
    temp = df
    
    for security in securities:
        temp = df[df["SecuritiesCode"] == security].copy()
        temp = temp.sort_values("Date")
        list_df.append(temp)
    
    #separating training and validation data
    train_ratio = 0.8
    
    train_df_list = []
    val_df_list = []
    
    for df in list_df:
        train_df_list.append(df.iloc[:int(df.shape[0]*train_ratio),:])
        val_df_list.append(df.iloc[int(df.shape[0]*train_ratio):,:])

    input_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    label = 'Target'

    input_train, label_train = df_to_ds(train_df_list[0], input_features, label)
    input_val, label_val = df_to_ds(val_df_list[0], input_features, label)

    print(input_train)
    print(label_train)

    #Initializing the RNN
    model = Sequential()
    #Making a robust stacked LSTM layer
    #Adding the first LSTM layer and some Dropout Regularization
    model.add(LSTM(units = 1, return_sequences = True, input_shape = (input_train.shape[1], 1) ) )
    model.add(Dropout(0.2))
    #Adding the second LSTM layer and some Dropout Regularization
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    #Adding the third LSTM layer and some Dropout Regularization
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    #Adding the fourth LSTM layer and some Dropout Regularization
    model.add(LSTM(units = 50, return_sequences = False))
    model.add(Dropout(0.2))

    #Adding the output layer
    model.add(Dense(units = 1))

    #Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics=['accuracy'])

    model.summary()
    plot_model(model, show_shapes=True, rankdir="LR")

    #Fitting the RNN to the Training set
    model.fit(input_train, label_train, epochs = 100)  
    model.save('stock_prediction_rnn.h5')