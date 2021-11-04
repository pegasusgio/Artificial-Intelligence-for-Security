import pandas
from pandas import read_csv


def loadCsv(path):
    return read_csv(path)


def preElaborationData(dataframe, columns):
    for c in columns:
        print('Column name: ', c)
        statistics = (dataframe[c].describe())
        print(statistics, '\n')
    return statistics


def removeColumns(dataframe, columns):
    removedColumns = []
    shape = dataframe.shape
    for c in columns:
        if dataframe[c].min() == dataframe[c].max():  # check if min and max values are equal
            removedColumns.append(c)
    dataframe = dataframe.drop(columns=removedColumns)
    print('Removed columns: ', removedColumns)
    print('Dim before the removal:', shape)
    print('Dim after the removal:', dataframe.shape)
    return dataframe, removedColumns
