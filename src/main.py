# file path
from functions import loadCsv, preElaborationData, removeColumns

# retrieve the csv file
path = "C:\\Users\\pegasusgio\\Downloads\\trainDdosLabelNumeric.csv"
dataframe = loadCsv(path)  # return a DataFrame
shape = dataframe.shape  # return a tuple representing the dimensionality of the DataFrame.

# print dataframe and others data
print('The training set observed by the csv file is the following:\n', dataframe)
print('The matrix size is: ', shape)
print('The first five rows:\n', dataframe.head())
print('The attributes labels:\n', dataframe.columns, '\n')

# pre-elaboration
columns = list(dataframe.columns.values)  # list take an Index type and return attributes' labels as array
statistics = preElaborationData(dataframe, columns)  # loop over each column and compute describe function
dataframe, removedColumns = removeColumns(dataframe, columns)  # remove the columns that have same value on min-max(not only 0, they must be equal)
print('The removed Columns are:', removedColumns)
