import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


def make_read_path(root, category, sub, clas, trial):
        """Generate a file path for raw data 
        """
        path = '{0}/{1}/sub{2}/class{3}trial{4}.csv'.\
                format(root, category, sub, clas, trial)

        return path

class DataManipulator:
    # def __init__(self, data_name, data_info, sub, trial_set, in_norm=True):
    #     data_name = data_name['data_name']
    #     n_class = data_info['n_class']
    #     preprocess = data_info['preprocess']
    def __init__(self, sub, trial_set, in_norm=True):
        data_name = 'furui_lab'
        n_class = 6
        n_length = 45
        n_dim = 4
        cutoff =1.0
        extraction_rate = 0.1
        down_sampling_rate = 1
        

        # data_x = np.zeros((1,preprocess['length'],preprocess['n_channel']))
        data_x = np.zeros((1,n_length,n_dim))
        data_y = np.zeros((1,1))

        # self.data = data
        # self.labels = labels
        # self.batch_size = batch_size
        # self.test_size = test_size
        # self.random_state = random_state

        n_trial = len(trial_set)

        for c in range(n_class):
            for t in range(n_trial):

                category = 'extracted/{:.1f}Hz'.format(cutoff)


                # data_path = f'../../data/{data_name}/{category}/sub{sub}/class{c+1}trial{t+1}.csv'
                data_path = f'data/{data_name}/{category}/sub{sub}/class{c+1}trial{t+1}.csv'
               
                # fname = make_read_path(data_name, category, sub+1, c+1, trial_set[t] + 1)

                read_data = np.loadtxt(data_path,usecols=[0,1,2,3],delimiter=',')
                
                read_data = read_data[int(len(read_data)*extraction_rate):]
                read_data = read_data[::down_sampling_rate]
                read_data = read_data.reshape(-1, n_length, n_dim)

                label = np.full(len(read_data), c, dtype='int8')
                label = label.reshape(-1,1)

                data_x = np.concatenate((data_x, read_data), axis=0)
                data_y = np.concatenate((data_y, label), axis=0)
        
        self.data_x = data_x[1:].astype(np.float64)
        self.data_y = data_y[1:].astype(np.int8)

    def set_dataset(self):
        data_x, data_y = torch.Tensor(self.data_x), torch.Tensor(self.data_y)
        data_x, data_y = data_x.to('cpu'), data_y.to('cpu')
        self.dataset = TensorDataset(data_x, data_y)

    def set_dataloader(self, batch_size):
        self.dataloader = DataLoader(self.dataset, batch_size=30, shuffle=True)



    

    # def concatenate_data(data_path, sub, cutoff, trial_set, extraction_rate=0, down_sampling_rate=1):
    #     """
    #     Concatenate data with class labels
    #     """
    #     _, _, n_channel, n_class, _ = config_parser(data_path)

    #     # Generate empty array
    #     stacked_data = np.empty((0, n_channel))
    #     class_label = np.empty(0)

    #     n_trial = len(trial_set)    

    #     # Data concatenation
    #     for c in range(n_class):
    #         stacked_data_trial = np.empty((0, n_channel))
    #         for t in range(n_trial):
    #             category = 'extracted/{:.1f}Hz'.format(cutoff)
    #             fname = make_read_path(data_path, category, sub+1, c+1, trial_set[t] + 1)
                
    #             read_data = pd.read_table(fname, header=None).values
    #             read_data = read_data[int(len(read_data)*extraction_rate):]
    #             read_data = read_data[::down_sampling_rate]

    #             # Stack data along with trial direction
    #             stacked_data_trial = np.vstack([stacked_data_trial, read_data])
            
            
    #         stacked_data = np.vstack([stacked_data, stacked_data_trial])
    #         class_label = np.hstack([class_label, np.full(len(stacked_data_trial), c, dtype='int8')])

    #     return stacked_data.astype(np.float64), class_label.astype(np.int8)

    # def split_data(self):
    #     X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=self.test_size, random_state=self.random_state)
    #     return X_train, X_test, y_train, y_test

    # def create_dataloaders(self):
    #     X_train, X_test, y_train, y_test = self.split_data()
    #     train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    #     test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    #     train_loader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
    #     test_loader = DataLoader(test_data, shuffle=True, batch_size=self.batch_size)
    #     return train_loader, test_loader

    # def get_data(self):
    #     return self.data, self.labels