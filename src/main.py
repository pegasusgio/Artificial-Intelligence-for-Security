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

# stratified K-fold cross validation
cols = list(dataframe.columns.values)  # retrieves all the attribute names
independentList = cols[0:dataframe.shape[1] - 1]  # remove from the cols list named 'Label'
print('Independent List:', independentList, '\n')
target = 'Label'
X = dataframe.loc[:, independentList]  # projection done on the independent variables
y = dataframe[target]  # projection done on the label
folds = 5
seed = 43  # value to randomize the random split
np.random.seed(seed)
xTrainList, xTestList, yTrainList, yTestList = stratifiedKFold(X, y, folds, seed)


# decision Tree
t = decisionTreeLearner(X, y, 'entropy', 0.001, seed)
showTree(t)

# Takes the 5-fold cross-validation to determine best configuration with respect to the criterion
bestCcp_alpha, bestCriterion, bestF1score = determineDecisionTreekFoldConfiguration(xTrainList,
                                                                                    xTestList,
                                                                                    yTrainList,
                                                                                    yTestList,
                                                                                    seed)
print('********************************')
print('bestCcp_alpha is:', bestCcp_alpha)
print('bestCriterion is:', bestCriterion)
print('bestF1score is:', bestF1score)

# decision Tree with best possible parameters
bestTree = decisionTreeLearner(X, y, bestCriterion, bestCcp_alpha, seed)
showTree(bestTree)

# Load the testing set testDdosLabelNumeric.csv and generate the predictions for the testing
# samples by using the decision trees learned from the entire training set with the best configuration
# retrieve the test dataset
path = "C:\\Users\\pegasusgio\\Downloads\\testDdosLabelNumeric.csv"
test = loadCsv(path)  # return a DataFrame
test = test.drop(columns=removedColumns)


# retrieve xTest and yTest from testSet
colsTest = list(test.columns.values)  # retrieves all the attribute names
independentListTest = cols[0:test.shape[1] - 1]  # remove from the cols list named 'Label'
xTest = test.loc[:, independentListTest]  # projection done on the independent variables
yTest = test[target]  # projection done on the label

# predict xTest and compute the f1score on the best possible configuration tree
f1score = decisionTreeF1(xTest, yTest, bestTree)
print('F1 Score for the testSet is: ', f1score)
