import numpy as np
from sklearn.cluster import KMeans

from functions import loadCsv, preElaborationData, removeColumns, printHistogram, stratifiedKFold,\
    decisionTreeLearner, showTree, assignClassToCluster, dataScaler, elbowMethod, decisionTreeF1,\
    computeConfusionMatrix, showClassificationReport, determineDecisionTreekFoldConfiguration

# retrieve the csv file
path = "/home/pegasusgio/Scaricati/trainDdosLabelNumeric.csv"
dataframe = loadCsv(path)  # return a DataFrame
shape = dataframe.shape  # return a tuple representing the dimensionality of the DataFrame.

# print dataframe and others data
print('The training set observed by the csv file is the following:\n', dataframe, '\n')
print('The matrix size is: ', shape, '\n')
print('The first five rows:\n', dataframe.head(), '\n')
print('The attributes labels:\n', dataframe.columns, '\n')

# pre-elaboration
columns = list(dataframe.columns.values)  # list take an Index type and return attributes' labels as array
preElaborationData(dataframe, columns)  # loop over each column and compute describe function
dataframe, removedColumns = removeColumns(dataframe, columns)  # remove the columns that have same value on min-max
print('The removed Columns are:', removedColumns, '\n')

# retrieve cardinality of each class and print a histogram
printHistogram(dataframe)

# stratified K-fold cross validation
cols = list(dataframe.columns.values)  # retrieves all the attribute names
independentList = cols[0:dataframe.shape[1] - 1]  # remove from the cols list named 'Label'
print('Independent List:', independentList, '\n')
target = 'Label'
X = dataframe.loc[:, independentList]  # projection of the original dataset on the independent attributes
y = dataframe[target]  # projection of the original dataset on the label
folds = 5
seed = 43  # value to randomize the random split (guarantee that the same result is achieved), this number controls
# the sequence of random number that are generated
np.random.seed(seed)  # Set the seed in numpy library to achieve the same random list after closing the program and
# re-run again
xTrainList, xTestList, yTrainList, yTestList = stratifiedKFold(X, y, folds, seed)

seed = 43
np.random.seed(seed)

# decision Tree
t = decisionTreeLearner(X, y, 'entropy', 0.001, seed)
showTree(t)

# Takes the 5-fold cross-validation to determine the best configuration with respect to the criterion and ccp_alpha
bestCcp_alpha, bestCriterion, bestF1score = determineDecisionTreekFoldConfiguration(xTrainList,
                                                                                    xTestList,
                                                                                    yTrainList,
                                                                                    yTestList,
                                                                                    seed)
print('********************************')
print('bestCcp_alpha is:', bestCcp_alpha)
print('bestCriterion is:', bestCriterion)
print('bestF1score is:', bestF1score)

# decision Tree with the best possible parameters
bestTree = decisionTreeLearner(X, y, bestCriterion, bestCcp_alpha, seed)
showTree(bestTree)

# Load the testing set testDdosLabelNumeric.csv and generate the predictions for the testing
# samples by using the decision trees learned from the entire training set with the best configuration
# retrieve the test dataset
path = "/home/pegasusgio/Scaricati/testDdosLabelNumeric.csv"
test = loadCsv(path)  # return a DataFrame
test = test.drop(columns=removedColumns)

# retrieve xTest and yTest from testSet
colsTest = list(test.columns.values)  # retrieves all the attribute names
independentListTest = cols[0:test.shape[1] - 1]  # remove from the cols list named 'Label'
xTest = test.loc[:, independentListTest]  # projection done on the independent variables
yTest = test[target]  # projection done on the label

# predict xTest and compute the f1score on the best possible configuration tree
f1score = decisionTreeF1(xTest, yTest, bestTree)
print('F1 Score for the testSet is: ', f1score, '\n')

# Determine and show the confusion matrix
yPred = bestTree.predict(xTest)
modelName = "Decision Tree"
computeConfusionMatrix(yTest, yPred, modelName)

# Determine and show the classification report
print('**************** CLASSIFICATION REPORT ****************')
showClassificationReport(yTest, yPred)

# ********************* Kmeans *********************

# 1step: remove the class (we already did that, the result is 'X') and scale the data
Xscaled = dataScaler(X)
print('The scaled data are the following: ')
print(Xscaled, '\n')

# 2step: Elbow method
elbowMethod(Xscaled, seed)

# after observing the elbow graph, run kmeans with the best k
bestK = 10
km = KMeans(n_clusters=bestK, random_state=seed)
km = km.fit(Xscaled)

# kmlabels is an array having the cluster label into which each example lies
kmLabels = km.labels_

#  assign_class_to_cluster
clusters, class_to_cluster, purity = assignClassToCluster(y, kmLabels)
print("clusters: ", clusters)
print("class_to_cluster: ", class_to_cluster)
print("purity: ", purity)

# Remove the class (we already did that, the result is 'xTest') and scale the data
xTestScaled = dataScaler(xTest)
print('The scaled data are the following: ')
print(xTestScaled, '\n')

# Predict the class of the testing sample based of the selected cluster
clustering_prediction_test = km.predict(xTestScaled)
prediction_test = [class_to_cluster[c] for c in clustering_prediction_test]

# compute and display the confusion matrix
modelName = "K-Means"
computeConfusionMatrix(yTest, prediction_test, modelName)
showClassificationReport(yTest, prediction_test)
