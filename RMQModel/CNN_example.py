# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import scale
from tensorflow.keras import backend as K, callbacks
from sklearn.metrics import accuracy_score as accuracy, f1_score, mean_absolute_error as mae
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.utils.vis_utils import plot_model



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


dataset_NASDAQ = pd.read_csv("/kaggle/input/cnnpred-stock-market-prediction/Processed_NASDAQ.csv", parse_dates=['Date'])
dataset_NYSE = pd.read_csv("/kaggle/input/cnnpred-stock-market-prediction/Processed_NYSE.csv", parse_dates=['Date'])
dataset_SP = pd.read_csv("/kaggle/input/cnnpred-stock-market-prediction/Processed_SP.csv", parse_dates=['Date'])
dataset_DJI = pd.read_csv("/kaggle/input/cnnpred-stock-market-prediction/Processed_DJI.csv", parse_dates=['Date'])
dataset_RUSSELL = pd.read_csv("/kaggle/input/cnnpred-stock-market-prediction/Processed_RUSSELL.csv", parse_dates=['Date'])


dataset_NASDAQ.index = dataset_NASDAQ['Date']
dataset_NYSE.index = dataset_NYSE['Date']
dataset_SP.index = dataset_SP['Date']
dataset_DJI.index = dataset_DJI['Date']
dataset_RUSSELL.index = dataset_RUSSELL['Date']

dataset_NASDAQ.columns

dataset_NASDAQ

whole_data = dataset_NASDAQ.append(dataset_NYSE, ignore_index=True)

whole_data

whole_data = whole_data.append(dataset_SP, ignore_index=True)
whole_data = whole_data.append(dataset_DJI, ignore_index=True)
whole_data = whole_data.append(dataset_RUSSELL, ignore_index=True)
whole_data

whole_data["Close"]

print(len(dataset_NASDAQ ))
print(len(dataset_NYSE))
print(len(dataset_SP))
print(len(dataset_DJI ))
print(len(dataset_RUSSELL))

whole_data.isnull().sum()

whole_data.info()

import matplotlib.pyplot as plt
whole_data.hist(figsize=(30,30))
plt.show()

# Kalan kısma sonra bak:
# Neyi predict edeceğiz ??

predict_index = 'DJI'
number_of_stocks = 0
order_stocks = []
predict_day = 1


def prepare_for_CNN():
    global number_of_stocks
    global samples_in_each_stock
    global number_feature
    # global predict_index
    global order_stocks
    tottal_train_data = np.empty((0, 82))
    tottal_train_target = np.empty((0))
    tottal_test_data = np.empty((0, 82))
    tottal_test_target = np.empty((0))

    for data in [dataset_DJI, dataset_NASDAQ, dataset_NYSE, dataset_RUSSELL, dataset_SP]:

        number_of_stocks += 1

        df_name = data['Name'][0]
        order_stocks.append(df_name)
        del data['Name']

        target = (data['Close'][predict_day:] / data['Close'][:-predict_day].values).astype(int)
        print(target)
        print("*****")
        data = data[:-predict_day]
        target.index = data.index
        # Becasue of using 200 days Moving Average as one of the features
        data = data[200:]
        data = data.fillna(0)
        data['target'] = target
        target = data['target']
        del data['target']
        del data['Date']
        # data['Date'] = data['Date'].apply(lambda x: x.weekday())

        number_feature = data.shape[1]
        samples_in_each_stock = data.shape[0]

        train_data = data[data.index < '2016-04-21']
        train_data = scale(train_data)

        if df_name == predict_index:
            tottal_train_target = target[target.index < '2016-04-21']
            tottal_test_target = target[target.index >= '2016-04-21']

        data = pd.DataFrame(scale(data.values), columns=data.columns)
        data.index = target.index
        test_data = data[data.index >= '2016-04-21']

        tottal_train_data = np.concatenate((tottal_train_data, train_data))
        print(tottal_train_data.shape)
        tottal_test_data = np.concatenate((tottal_test_data, test_data))
        print(tottal_test_data.shape)

    train_size = int(tottal_train_data.shape[0] / number_of_stocks)
    print("Train size:", train_size)
    test_size = int(tottal_test_data.shape[0] / number_of_stocks)
    print("Test size:", test_size)

    tottal_train_data = tottal_train_data.reshape(number_of_stocks, train_size, number_feature)
    print("Total train data shape:", tottal_train_data.shape)
    tottal_test_data = tottal_test_data.reshape(number_of_stocks, test_size, number_feature)
    print("Total test data shape:", tottal_test_data.shape)

    return tottal_train_data, tottal_test_data, tottal_train_target, tottal_test_target


def cnn_data_sequence(data, target, seque_len):
    print ('sequencing data ...')
    new_train = []
    new_target = []

    for index in range(data.shape[1] - seque_len + 1):
        new_train.append(data[:, index: index + seque_len])
        new_target.append(target[index + seque_len - 1])

    new_train = np.array(new_train)
    new_target = np.array(new_target)

    return new_train, new_target

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision_pos = precision(y_true, y_pred)
    recall_pos = recall(y_true, y_pred)
    precision_neg = precision((K.ones_like(y_true)-y_true), (K.ones_like(y_pred)-K.clip(y_pred, 0, 1)))
    recall_neg = recall((K.ones_like(y_true)-y_true), (K.ones_like(y_pred)-K.clip(y_pred, 0, 1)))
    f_posit = 2*((precision_pos*recall_pos)/(precision_pos+recall_pos+K.epsilon()))
    f_neg = 2 * ((precision_neg * recall_neg) / (precision_neg + recall_neg + K.epsilon()))

    return (f_posit + f_neg) / 2

def sklearn_acc(model, test_data, test_target):
    overall_results = model.predict(test_data)
    test_pred = (overall_results > 0.5).astype(int)
    acc_results = [mae(overall_results, test_target), accuracy(test_pred, test_target),
                   f1_score(test_pred, test_target, average='macro')]

    return acc_results


number_filter = [8, 8, 8]


def CNN(train_data, test_data, train_target, test_target):
    # hisory of data in each sample
    seq_len = 60
    epoc = 100
    drop = 0.1

    cnn_train_data, cnn_train_target = cnn_data_sequence(train_data, train_target, seq_len)
    cnn_test_data, cnn_test_target = cnn_data_sequence(test_data, test_target, seq_len)
    result = []

    for i in range(1, 5):
        K.clear_session()
        print('i: ', i)

        print('fitting model')

        model = Sequential()

        # layer 1
        model.add(
            Conv2D(number_filter[0], (1, 1), activation='relu', input_shape=(number_of_stocks, seq_len, number_feature),
                   data_format='channels_last'))
        model.add(Dropout(0.1))  # added

        # layer 2
        model.add(BatchNormalization())  # added
        model.add(Conv2D(number_filter[1], (number_of_stocks, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(1, 2)))
        model.add(Dropout(0.2))  # added

        # layer 3
        model.add(Conv2D(number_filter[2], (1, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(1, 2)))

        # Flattening Layer:
        model.add(Flatten())
        model.add(Dropout(0.4))  # added

        # Last Layer:
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='Adam', loss='mae', metrics=['acc', f1])

        # best_model = callbacks.ModelCheckpoint(filepath, monitor='val_f1', verbose=0, save_best_only=True,
        # save_weights_only=False, mode='max', period=1)

        model.fit(cnn_train_data, cnn_train_target, epochs=epoc, batch_size=128, verbose=0,
                  validation_split=0.25)  # callbacks=[best_model],

        #   model = load_model(filepath, custom_objects={'f1': f1})
        test_pred = sklearn_acc(model, cnn_test_data, cnn_test_target)
        print(test_pred)
        result.append(test_pred)

        model.summary()  # added

        # plot_model(model)

    print('saving results')
    results = pd.DataFrame(result, columns=['MAE', 'Accuracy', 'F-score'])
    results = results.append([results.mean(), results.max(), results.std()], ignore_index=True)
    # results.to_csv(join(Base_dir, '3D-models/{}/new results.csv'.format(predict_index)), index=False)
    return results


train_data, test_data, train_target, test_target = prepare_for_CNN()

CNN(train_data, test_data, train_target, test_target)