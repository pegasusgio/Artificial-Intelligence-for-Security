# file path
import numpy as np

from functions import loadCsv, preElaborationData, removeColumns, printHistogram, stratifiedKFold, decisionTreeLearner, \
    showTree

# retrieve the csv file
path = "C:\\Users\\pegasusgio\\Downloads\\trainDdosLabelNumeric.csv"
dataframe = loadCsv(path)  # return a DataFrame
shape = dataframe.shape  # return a tuple representing the dimensionality of the DataFrame.

# print dataframe and others data
print('The training set observed by the csv file is the following:\n', dataframe, '\n')
print('The matrix size is: ', shape, '\n')
print('The first five rows:\n', dataframe.head(), '\n')
print('The attributes labels:\n', dataframe.columns, '\n')

# pre-elaboration
columns = list(dataframe.columns.values)  # list take an Index type and return attributes' labels as array
statistics = preElaborationData(dataframe, columns)  # loop over each column and compute describe function
dataframe, removedColumns = removeColumns(dataframe, columns)  # remove the columns that have same value on min-max
print('The removed Columns are:', removedColumns, '\n')

# retrieve cardinality of each class and print an histogram
printHistogram(dataframe)

# stratified K-fold CV
cols = list(dataframe.columns.values)   # retrieves all the attribute names
independentList = cols[0:dataframe.shape[1]-1]   # remove from the cols list named 'Label'
print('Independent List:', independentList, '\n')
target = 'Label'
X = dataframe.loc[:, independentList]  # all dataframe values except for 'Label'
y = dataframe[target]  # array containing the only label values
folds = 5  # number of folds
seed = 42  # value to randomize the random split
np.random.seed(seed)
xTrainList, xTestList, yTrainList, yTestList = stratifiedKFold(X, y, folds, seed)
print('\n', 'The first 5 rows')
print(dataframe.head())


# decision Tree
'''
t = decisionTreeLearner(X, y, 'entropy', 0.001, seed)
showTree(t)
'''