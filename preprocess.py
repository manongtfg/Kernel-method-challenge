import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def preprocess(file1, file2, file3, train=True, numeric_data=False,labels_1=None, labels_2=None, labels_3=None):
    """ Download all necessary files for the training set and the testing set. 
    If train is set to False, no labels will be download."""    

    #If we use DNA sequences directly 
    if not numeric_data:
        # We download training set
        df_1 = pd.read_csv(file1, sep=",").reset_index(drop=True)
        df_2 = pd.read_csv(file2, sep=",").reset_index(drop=True)
        df_3 = pd.read_csv(file3, sep=",").reset_index(drop=True)

        if train:
            labels_1 = pd.read_csv(labels_1)
            labels_1_train = labels_1.loc[:, 'Bound'].to_numpy()

            labels_2 = pd.read_csv(labels_2)
            labels_2_train = labels_2.loc[:, 'Bound'].to_numpy()

            labels_3 = pd.read_csv(labels_3)
            labels_3_train = labels_3.loc[:, 'Bound'].to_numpy()

            #Encoding labels
            labels_train = np.concatenate((labels_1_train, labels_2_train, labels_3_train))
            encoder = OneHotEncoder(sparse_output=False)
            labels__train_onehot = encoder.fit_transform(labels_train.reshape(-1, 1))


        # Concatenation 
        df = pd.concat([df_1, df_2, df_3], axis=0)
        if train:
            return df, labels_train, labels__train_onehot
        
        return df
    
    #If we use the encoded sequences
    else:
        # We download training set 
        df_1 = pd.read_csv(file1, sep=" ", header=None)
        df_2 = pd.read_csv(file2, sep=" ", header=None)
        df_3 = pd.read_csv(file3, sep=" ", header=None)

        if train:
            labels_1 = pd.read_csv(labels_1)
            labels_1_train = labels_1.loc[:, 'Bound'].to_numpy()

            labels_2 = pd.read_csv(labels_2)
            labels_2_train = labels_2.loc[:, 'Bound'].to_numpy()

            labels_3 = pd.read_csv(labels_3)
            labels_3_train = labels_3.loc[:, 'Bound'].to_numpy()

            #Encoding labels
            labels_train = np.concatenate((labels_1_train, labels_2_train, labels_3_train))
            encoder = OneHotEncoder(sparse_output=False)
            labels__train_onehot = encoder.fit_transform(labels_train.reshape(-1, 1))


        # Concatenation 
        df = pd.concat([df_1, df_2, df_3], axis=0)
        if train:
            return df, labels_train, labels__train_onehot
        
        return df



