import numpy as np
import pandas as pd
import time

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


    

