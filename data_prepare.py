import os
import numpy as np
import pandas as pd
import urllib

def download_data(data_path):
    source_url = "http://r7-yosou.hippy.jp/T-data.xls"
    xls_source = urllib.request.urlopen(source_url)
    with open(data_path, "wb") as xls_file:
        xls_file.write(xls_source.read())

def xls2array():
    filename = "T-data.xls"
    xls_file = pd.ExcelFile(filename)
    df_list = xls_file.parse("N4")
    df_array = df_list.values
    return df_array

def numbers2onehotvector(numbers):
    onehotvector = np.zeros(40, dtype="int32")
    for i in range(4):
        onehotvector[i*10 + numbers[i]] = 1
    return onehotvector

def onehotvector2numbers(onehotvector):
    numbers =  []
    for i in range(4):
        i = i * 10
        digit_onehot = onehotvector[i:i+10]
        number = np.where(digit_onehot == 1)
        numbers.append(number[0])
    numbers = np.array(numbers)
    return numbers


if __name__ == "__main__":
    data_path = os.path.join("T-data.xls") 
    # download_data(data_path)
    numbers_array = xls2array()
    numbers_array = numbers_array[:, :4]
    onehotvector = numbers2onehotvector(numbers_array[-1])
    numbers = onehotvector2numbers(onehotvector)
    print(numbers)
