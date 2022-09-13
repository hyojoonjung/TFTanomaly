import numpy as np
import pandas as pd

def stride_series(data, example_length, stride, train=True, raw_data=None, dataset=None):
    series = []
    for i in range(0,len(data) - example_length, stride):
        series.append(data[i: (i + example_length)])
        # print(len(data[i: (i + example_length)]))
    for i in range(len(series)):
        if len(series[i]) !=  example_length:
            print(len(series[i]))
    # series = np.array(series)
    # print(series)
    if train is True:
        return series

    else:
        if not isinstance(raw_data, pd.DataFrame):
            raise ValueError("'raw_data' expected <class 'pandas.core.frame.DataFrame'>, got '{}'".format(type(raw_data)))

        labels = np.array([0 for i in range(len(series))])
        if dataset == 'GHL':
            ## GHL
            dangers = np.array(raw_data.index[raw_data['DANGER'] == 1])
            faults = np.array(raw_data.index[raw_data['FAULT'] == 1])
            labels[dangers] = 1
            labels[faults] = 1
        elif dataset == 'WADI':
            ## WADI
            attack = np.array(raw_data.index[raw_data['Attack'] == -1])
            labels[attack] = 1
        elif dataset == 'SMD':
            ## SMD
            attack = np.array(raw_data.index[raw_data['label'] == 1])
            labels[attack] = 1
        
        return series, labels 

def standard_scaler(data, mu_sigma_values):
    data_scaled = data.values.copy()
    col = data.columns
    # print(col)
    if len(mu_sigma_values) == 0:
        #data.shape[1] = num_column
        for i in range(data.shape[1]):
            values = data_scaled[:, i]
            mean = np.mean(values)
            if np.std(values) == 0:
                std = 1
            else:
                std = np.std(values)

            mu_sigma_values.append([mean, std])

            values_scaled = (values - mean) / std
            data_scaled[:, i] = values_scaled
    else:
        for i in range(data.shape[1]):
            values = data_scaled[:, i]

            values_scaled = (values - mu_sigma_values[i][0]) / mu_sigma_values[i][1]
            data_scaled[:, i] = values_scaled
    data_scaled = pd.DataFrame(data_scaled)
    data_scaled.columns = col
    # print(data_scaled.head())
    return data_scaled, mu_sigma_values

def min_max_scaler(data, min_max_values):
    data_scaled = data.values.copy()
    col = data.columns
    
    if len(min_max_values) == 0:
        for i in range(data.shape[1]):
            values = data_scaled[:, i]
            min_max_values.append([np.nanmin(values), np.nanmax(values)])
            
            if not all(values == 0):
                if np.nanmax(values) != np.nanmin(values):
                    values_scaled = (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values))
                else:
                    values_scaled = values
                data_scaled[:, i] = values_scaled
            else:
                min_max_values.append([0, 0])

    else:
        for i in range(data.shape[1]):
            values = data[:, i]

            if not all(values == 0):
                values_scaled = (values - min_max_values[i][0]) / (min_max_values[i][1] - min_max_values[i][0])
                data_scaled[:, i] = values_scaled

    data_scaled = pd.DataFrame(data_scaled)
    data_scaled.columns = col
    return data_scaled, min_max_values