from pandas import read_csv, DataFrame
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier, plot_tree
import collections
import matplotlib.pyplot as plt


def loadCsv(path):
    return read_csv(path)


def preElaborationData(dataframe, columns):
    statistics = DataFrame()
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
    print('Dim after the removal:', dataframe.shape, '\n')
    return dataframe, removedColumns


def countLabels(dataframe):
    return collections.Counter(dataframe['Label'])
    # return dataframe['Label'].value_counts()


def printHistogram(dataframe):
    c = countLabels(dataframe)
    labels = (list(c))
    plt.hist(dataframe['Label'], len(labels))
    plt.show()
    return


def stratifiedKFold(X, y, folds, seed):
    # This cross - validation object is a variation of KFold that return the stratified folds.
    # The fold are made by preserving the percentage of samples for each class.
    skf = model_selection.StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)

    # empty lists declaration
    xTrainList = []
    xTestList = []
    yTrainList = []
    yTestList = []

    # looping over split
    for trainIndex, testIndex in skf.split(X, y):
        print("TRAIN:", trainIndex, "TEST:", testIndex)
        xTrainList.append(X.iloc[trainIndex])
        xTestList.append(X.iloc[testIndex])
        yTrainList.append(y.iloc[trainIndex])
        yTestList.append(y.iloc[testIndex])
    return xTrainList, xTestList, yTrainList, yTestList


def decisionTreeLearner(X, y, criterion, ccp_alpha, seed):
    tree = DecisionTreeClassifier(criterion=criterion, random_state=seed, ccp_alpha=ccp_alpha)
    tree.fit(X, y)
    return tree


def showTree(tree):
    plt.figure(figsize=(40, 30))
    plot_tree(tree)
    plt.show()
