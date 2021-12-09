import numpy as np
from pandas import read_csv
from sklearn import model_selection
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
import collections
import matplotlib.pyplot as plt


def loadCsv(path):
    return read_csv(path)


def preElaborationData(dataframe, columns):
    for c in columns:
        print('Column name: ', c)
        statistics = (dataframe[c].describe())
        print(statistics, '\n')


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


def decisionTreeF1(xTest, yTest, tree):
    yPred = tree.predict(xTest)
    f1score = f1_score(yTest, yPred, average='weighted')
    return f1score


def determineDecisionTreekFoldConfiguration(xTrainList, xTestList, yTrainList, yTestList, seed):
    # variables declaration
    criterionList = ['entropy', 'gini']
    bestCcp_alpha = 0
    bestCriterion = ''
    bestF1score = 0
    minRange = 0
    maxRange = 0.05
    step = 0.001
    counter = 0

    for x, y, z, w in zip(xTrainList, yTrainList, xTestList, yTestList):
        counter = counter + 1
        for ccp_alpha in np.arange(minRange, maxRange, step):
            for criterion in criterionList:
                t = decisionTreeLearner(x, y, criterion, ccp_alpha, seed)
                f1score = decisionTreeF1(z, w, t)
                print('***************************')
                print('Iteration:', counter)
                print('Ccp_alpha:', ccp_alpha)
                print('Criterion:', criterion)
                print('Seed:', seed)
                print('f1score:', f1score)
                if f1score > bestF1score:
                    bestF1score = f1score
                    bestCriterion = criterion
                    bestCcp_alpha = ccp_alpha
    return bestCcp_alpha, bestCriterion, bestF1score


def computeConfusionMatrix(yTest, yPred, tree):
    cm = confusion_matrix(yTest, yPred, labels=tree.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tree.classes_)
    disp.plot()
    plt.show()


def showClassificationReport(yTest, yPred):
    print(classification_report(yTest, yPred, labels=[0, 1, 2, 3, 4]))


def dataScaler(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    Xscaled = scaler.transform(X)
    return Xscaled


def elbowMethod(Xscaled):
    SumOfSquaredDistances = []
    allKmeans = []
    minRange = 2
    maxRange = 15
    K = range(minRange, maxRange)
    for i in K:
        km = KMeans(n_clusters=i)
        km = km.fit(Xscaled)
        # append inertia for each cluster in the array SumOfSquaredDistances
        SumOfSquaredDistances.append(km.inertia_)
        # save the km-th instance
        allKmeans.insert(i, km)

    # plot the results
    plt.plot(K, SumOfSquaredDistances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    return km.labels_


def assign_class_to_cluster(y, kmeans_labels):
    clusters = set(kmeans_labels)  # remove the duplicates => 0,1,2,...k, e.g cluster cardinality
    classes = set(y)  # remove the duplicates => 0,1,2,3,4, e.g the labels
    class_to_cluster = []
    N = 0
    purity = 0
    for c in clusters:  # loop over the cluster cardinality
        # create an array of element, each element is the index of the c-th element of the cluster label
        # (c=0, quindi cerca in kmeans_labels tutti i valori 0, e ne salva gli indici in indices)
        indices = [i for i in range(len(kmeans_labels)) if kmeans_labels[i] == c]

        # take all the prediction for the c-th cluster
        selected_classes = [y[i] for i in indices]
        max_class = -1
        max_PCF = -1
        for cl in classes:
            pcf = selected_classes.count(cl)
            N = N + pcf
            print('cluster ', c, ' class ', cl, ' pcf ', pcf)
            if pcf > max_PCF:
                max_PCF = pcf  # predicted class frequency
                max_class = cl
        # max_class is the class associated to the cluster c
        purity = purity + max_PCF
        class_to_cluster.append(max_class)
    return clusters, class_to_cluster, purity / N
