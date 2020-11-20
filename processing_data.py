import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import os
from dataset import WholeDataset



def load_data(file_fir):
    try:
        df_raw = pd.read_csv(file_fir, index_col='Date')
    except IOError:
        print("IO ERROR")
    return df_raw


def costruct_data_warehouse(ROOT_PATH, file_names, predict_day, seq_len):
    number_of_stocks = 0
    data_warehouse = {}

    for stock_file_name in file_names:

        file_dir = os.path.join(ROOT_PATH, stock_file_name)
        ## Loading Data
        try:
            df_raw = load_data(file_dir)
        except ValueError:
            print("Couldn't Read {} file".format(file_dir))

        number_of_stocks += 1

        data = df_raw
        df_name = data['Name'][0]
        del data['Name']

        target = (data['Close'][predict_day:] / data['Close'][:-predict_day].values).astype(int)
        data = data[:-predict_day]
        target.index = data.index
        # Becasue of using 200 days Moving Average as one of the features
        data = data[200:]
        data = data.fillna(0)
        data['target'] = target
        target = data['target']

        del data['target']

        number_feature = data.shape[1]
        samples_in_each_stock = data.shape[0]

        train_data = data[data.index < '2016-04-21']
        train_data1 = scale(train_data)

        train_target1 = target[target.index < '2016-04-21']
        train_data = train_data1[:int(0.75 * train_data1.shape[0])]
        train_target = train_target1[:int(0.75 * train_target1.shape[0])]

        valid_data = scale(train_data1[int(0.75 * train_data1.shape[0]) - seq_len:])
        valid_target = train_target1[int(0.75 * train_target1.shape[0]) - seq_len:]

        data = pd.DataFrame(scale(data.values), columns=data.columns)
        data.index = target.index
        test_data = data[data.index >= '2016-04-21']
        test_target = target[target.index >= '2016-04-21']

        data_warehouse[df_name] = [train_data, train_target, np.array(test_data), np.array(test_target), valid_data,
                                   valid_target]

    return data_warehouse, number_of_stocks, number_feature, samples_in_each_stock


def cnn_data_sequence_separately(tottal_data, tottal_target, data, target, seque_len):
    for index in range(data.shape[0] - seque_len + 1):
        tottal_data.append(data[index: index + seque_len])
        tottal_target.append(target[index + seque_len - 1])

    return tottal_data, tottal_target


def cnn_data_sequence(data_warehouse, seq_len):
    tottal_train_data = []
    tottal_train_target = []
    tottal_valid_data = []
    tottal_valid_target = []
    tottal_test_data = []
    tottal_test_target = []

    for key, value in data_warehouse.items():
        tottal_train_data, tottal_train_target = cnn_data_sequence_separately(tottal_train_data, tottal_train_target,
                                                                              value[0], value[1], seq_len)
        tottal_test_data, tottal_test_target = cnn_data_sequence_separately(tottal_test_data, tottal_test_target,
                                                                            value[2], value[3], seq_len)
        tottal_valid_data, tottal_valid_target = cnn_data_sequence_separately(tottal_valid_data, tottal_valid_target,
                                                                              value[4], value[5], seq_len)

    tottal_train_data = np.array(tottal_train_data)
    tottal_train_target = np.array(tottal_train_target)
    tottal_test_data = np.array(tottal_test_data)
    tottal_test_target = np.array(tottal_test_target)
    tottal_valid_data = np.array(tottal_valid_data)
    tottal_valid_target = np.array(tottal_valid_target)

    return tottal_train_data, tottal_train_target, tottal_test_data, tottal_test_target, tottal_valid_data, tottal_valid_target

def transforming_data_warehouse(data_warehouse, order_stocks, seq_len):
    transformed_data_loader = {}
    for name in order_stocks:
        value = data_warehouse[name]
        train = WholeDataset(*cnn_data_sequence_pre_train(value[0], value[1], seq_len))
        test = WholeDataset(*cnn_data_sequence_pre_train(value[2], value[3], seq_len))
        validation = WholeDataset(*cnn_data_sequence_pre_train(value[4], value[5], seq_len))

        transformed_data_loader[name] = [train, test, validation]

    return transformed_data_loader


def cnn_data_sequence_pre_train(data, target, seque_len):
    new_data = []
    new_target = []
    for index in range(data.shape[0] - seque_len + 1):
        new_data.append(data[index: index + seque_len])
        new_target.append(target[index + seque_len - 1])

    new_data = np.array(new_data)
    new_target = np.array(new_target)

    return new_data, new_target
