import os
import pandas as pd
from sklearn.model_selection import train_test_split


def prep_data():
    top_level_folder = os.getcwd() + '\COVID-19_Radiography_Dataset'

    # create classification list based on the folder names
    classes = ['COVID-19', 'NORMAL', 'Viral Pneumonia']

    # store paths and associated label
    data = []

    for ID, label in enumerate(classes):
        for file in os.listdir(top_level_folder+"/"+label+"/images"):
            data.append([os.path.join(label,file), label, os.path.join(top_level_folder,label,file)])

    data = pd.DataFrame(data,columns=['Image_file', 'Classification','path'])

    data.head()

    print('Number of Duplicated Samples: %d' % (data.duplicated().sum()))
    print('Number of Total Samples: %d' % (data.isnull().value_counts()))

    return data

def split_data(df):
    strat = df['Classification']
    # create 70% train and 30% dummy datasets
    train_df, dummy_df = train_test_split(df, train_size=0.7, shuffle=True, random_state=123, stratify=strat)

    strat = dummy_df['Classification']
    # create 50% validation and 50% test datasets (each 15%)
    valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=123, stratify=strat)

    # train_df.head()
    print(len(train_df))

    print(len(dummy_df))

    # valid_df.head()
    print(len(valid_df))

    # test_df.head()
    print(len(test_df))

    return train_df, dummy_df, valid_df, test_df
