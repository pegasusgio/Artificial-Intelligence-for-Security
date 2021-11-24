# file path
import numpy as np

from functions import loadCsv, preElaborationData, removeColumns, printHistogram, stratifiedKFold, decisionTreeLearner, \
    showTree, decisionTreeF1, determineDecisionTreekFoldConfiguration

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
cols = list(dataframe.columns.values)  # retrieves all the attribute names
independentList = cols[0:dataframe.shape[1] - 1]  # remove from the cols list named 'Label'
print('Independent List:', independentList, '\n')
target = 'Label'
X = dataframe.loc[:, independentList]  # projection done on the independent variables
y = dataframe[target]  # projection done on the label
folds = 5  # number of folds
seed = 43  # value to randomize the random split
np.random.seed(seed)
xTrainList, xTestList, yTrainList, yTestList = stratifiedKFold(X, y, folds, seed)
print('\n', 'The first 5 rows')
print(dataframe.head())

print('\n//////////////////////////////\n')
print('xTestList', '\n', xTestList, '\n')

# decision Tree
t = decisionTreeLearner(X, y, 'entropy', 0.001, seed)
showTree(t)

# retrieve the test dataset
path = "C:\\Users\\pegasusgio\\Downloads\\testDdosLabelNumeric.csv"
test = loadCsv(path)  # return a DataFrame
print("Test dimension before the drop:", '\n', test.shape)
test = test.drop(columns=removedColumns)
print("Test dimension after the drop:", '\n', test.shape)

# retrieve xTest and yTest
cols = list(test.columns.values)  # retrieves all the attribute names
independentList = cols[0:test.shape[1] - 1]  # remove from the cols list named 'Label'
print('Independent List:', independentList, '\n')
target = 'Label'
xTest = test.loc[:, independentList]  # projection done on the independent variables
yTest = test[target]  # projection done on the label
print('The xTest is:\n', xTest, '\n')
print('The yTest is:\n', yTest, '\n')
f1score = decisionTreeF1(xTest, yTest, t)
print('F1 Score is: ', f1score)  # f1score without k-fold cross validation

bestCcp_alpha, bestCriterion, bestF1score = determineDecisionTreekFoldConfiguration(xTrainList, xTestList, yTrainList,
                                                                                    yTestList, seed)
print('\n********************************')
print('bestCcp_alpha is:', bestCcp_alpha)
print('bestCriterion is:', bestCriterion)
print('bestF1score is:', bestF1score)
