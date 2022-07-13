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
    """
    Retrieve the csv file as Dataframe object
    :param path: Path where the csv file lies
    :return: the Dataframe object
    """
    return read_csv(path)


def preElaborationData(dataframe, columns):
    """
    Generate and print descriptive statistics for each attribute
    :param dataframe: The dataset used
    :param columns: The attributes' labels
    :return:
    """
    for c in columns:
        print('Column name: ', c)
        statistics = (dataframe[c].describe())
        print(statistics, '\n')


def removeColumns(dataframe, columns):
    """
    Remove the columns that have same value on min-max
    :param dataframe: The dataset used
    :param columns: The attributes' labels
    :return: The dataset used and the removed columns
    """
    removedColumns = []
    shape = dataframe.shape
    for c in columns:
        if dataframe[c].min() == dataframe[c].max():  # check if min and max values are equal
            removedColumns.append(c)
    dataframe.drop(columns=removedColumns, inplace=True)
    print('Removed columns: ', removedColumns)
    print('Dim before the removal:', shape)
    print('Dim after the removal:', dataframe.shape, '\n')
    return dataframe, removedColumns


def countLabels(dataframe):
    """
    Retrieve the labels values
    :param dataframe: The dataset used
    :return: The labels values
    """
    return collections.Counter(dataframe['Label'])


def printHistogram(dataframe):
    """
    Retrieve the cardinality of each class and print a histogram
    :param dataframe: The dataset used
    :return:
    """
    c = countLabels(dataframe)
    labels = (list(c))
    plt.hist(dataframe['Label'], len(labels))
    plt.title("Cardinality of each class")
    plt.show()


def stratifiedKFold(X, y, folds, seed):
    """
    This cross - validation object is a variation of KFold that return the stratified folds : the folds are made by
    preserving the percentage of samples for each class. If you work with shuffle = false, the dataset is divided by
    folds simply following the order of the example, but this behaviour can be risky, due to the fact that usually
    the class category in the datasets are organized sequentially. It's important because without it, you risk that
    one fold contain examples of the same class. Using Shuffle = true, instead,  the dataset is really randomly
    reorganized.
    :param X: Projection of the original dataset on the independent attributes
    :param y: Projection of the original dataset on the label
    :param folds: number of folds to generate
    :param seed: Value to randomize the random split (guarantee that the same result is achieved), this number controls
                 the sequence of random number that are generated
    :return: Four arrays containing all the 5 trials generated
    """

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
    """
    Learns a Decision Tree Classifier with the given parameter.
    Observation on parameters: ccp_alpha is interesting for minimal cost complexity pruning, that is a pruning
    strategy implemented in SKlearn for decision tree. Pruning strategy used to prune the decision tree. Is important
    when we try to construct the tree we fix the random_state to seed to be sure that also executing the algorithm
    several times, you construct the same tree. The randomic characteristic is a feature of this implementation not a
    feature of the algorithm
    :param X: Projection of the training set on the independent attributes
    :param y: Projection of the training set on the label
    :param criterion: The function to measure the quality of a split. Supported criteria are “gini” for the Gini
                      purity and “log_loss” and “entropy” both for the Shannon information gain
    :param ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost
                      complexity that is smaller than ccp_alpha will be chosen
    :param seed: Controls the randomness of the estimator.
    :return: The Decision Tree Classifier learned.
    """
    tree = DecisionTreeClassifier(criterion=criterion,
                                  random_state=seed,
                                  ccp_alpha=ccp_alpha)
    tree.fit(X, y)
    return tree


def showTree(tree):
    """
    Show the tree given as parameter. Computes and prints the number of nodes and leaves
    :param tree: The tree to be shown.
    :return:
    """
    plt.figure(figsize=(40, 30))
    plot_tree(tree, filled=True, fontsize=8, proportion=True)
    plt.show()
    nNodes = tree.tree_.node_count
    nLeaves = tree.tree_.n_leaves
    print("\nThe tree has ", nNodes, "nodes and ", nLeaves, " leaves.\n")


def decisionTreeF1(xTest, yTest, tree):
    """
    Execute the classification task and compute the weighted F1 score
    :param xTest: Projection of the test set on the independent attributes
    :param yTest: Projection of the test set on the label
    :param tree: The Decision Tree used to predict
    :return: The weighted F1 score
    """
    yPred = tree.predict(xTest)
    f1score = f1_score(yTest, yPred, average='weighted')
    return f1score


def determineDecisionTreekFoldConfiguration(xTrainList, xTestList, yTrainList, yTestList, seed):
    """
    Takes as input the 5-fold cross-validation to determine the best configuration with respect to the criterion (gini
    or entropy) and ccp_alpha (ranging among 0 and 0.05 with step 0.001). The best configuration is determined
    with respect to the weighted F1
    :param xTrainList: The 5 trials of the independent attributes on the training set
    :param xTestList: The 5 trials of the independent attributes on the test set
    :param yTrainList: The 5 trials of the target on the training set
    :param yTestList: The 5 trials of the target on the test set
    :param seed: Controls the randomness of the estimator.
    :return: Criterion, ccp_alpha and best weighted F1 of the best configuration.
    """

    criterionList = ['entropy', 'gini']
    bestCcp_alpha = 0
    bestCriterion = ''
    bestF1score = 0
    minRange = 0
    maxRange = 0.050
    step = 0.001
    counter = 0

    for ccp_alpha in np.arange(minRange, maxRange, step):
        for criterion in criterionList:
            f1Values = []
            for x, y, z, w in zip(xTrainList, yTrainList, xTestList, yTestList):
                counter = counter + 1
                t = decisionTreeLearner(x, y, criterion, ccp_alpha, seed)
                f1score = decisionTreeF1(z, w, t)
                f1Values.append(f1score)

                print('***************************')
                print('Iteration:', counter)
                print('Ccp_alpha:', ccp_alpha)
                print('Criterion:', criterion)
                print('Seed:', seed)
                print('f1score:', f1score)
                print('f1Values:', f1Values)

            avgF1 = np.mean(f1Values)
            print('avgF1:', avgF1)
            if avgF1 > bestF1score:
                bestF1score = avgF1
                bestCriterion = criterion
                bestCcp_alpha = ccp_alpha
    return bestCcp_alpha, bestCriterion, bestF1score


def computeConfusionMatrix(yTest, yPred, modelName):
    """
    Compute the Confusion Matrix on the model given as parameter
    :param modelName: Name of the model
    :param yTest: Projection of the test set on the label
    :param yPred: Prediction of the examples.
    :return:
    """
    cm = confusion_matrix(yTest, yPred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix for ' + modelName)
    plt.show()


def showClassificationReport(yTest, yPred):
    """
    Computes and shows the classification report
    :param yTest: Projection of the test set on the label
    :param yPred: Prediction of the examples.
    :return:
    """
    print(classification_report(yTest, yPred, labels=[0, 1, 2, 3, 4]))


def dataScaler(X):
    """
    Scale the data given as parameter
    :param X: The data to be scaled
    :return: The scaled data
    """
    scaler = MinMaxScaler()
    scaler.fit(X)
    Xscaled = scaler.transform(X)
    return Xscaled


def elbowMethod(Xscaled, seed):
    """
    Uses the elbow method to choose the best k for the clustering, according to the inertia. The k range lies between 2
    and 15. Lastly, this function plot the results on a graph.
    The elbow method is a heuristic used in determining the number of clusters in a data set. The method consists
    of plotting the explained variation as a function of the number of clusters and picking the elbow of the curve
    as the number of clusters to use.
    Inertia is the sum of squared distances of samples to their closest cluster center. The more clusters you use,
    the lower the inertia value will be
    :param Xscaled: The scaled data obtained by the function dataScaler
    :param seed: Controls the randomness of the estimator.
    :return:
    """
    SumOfSquaredDistances = []
    allKmeans = []
    minRange = 2
    maxRange = 15
    K = range(minRange, maxRange)
    for i in K:
        print('Computing Kmeans with K=', i)
        km = KMeans(n_clusters=i, random_state=seed)
        km = km.fit(Xscaled)
        # append inertia for each cluster in the array SumOfSquaredDistances
        SumOfSquaredDistances.append(km.inertia_)
        # save the km-th instance
        allKmeans.insert(i, km)

    # plot the results
    plt.plot(K, SumOfSquaredDistances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.show()


def assignClassToCluster(y, kmeansLabels):
    """
    Assign each cluster computed on the training set to a class (based on the purity)
    :param y: Projection of the training set on the label
    :param kmeansLabels: It's an array having the cluster label into which each example lies
    :return: The clusters, an array containing the class assigned to each cluster and the purity
    """
    clusters = set(kmeansLabels)  # remove the duplicates => 0,1,2,...k, e.g cluster cardinality
    classes = set(y)  # remove the duplicates => 0,1,2,3,4, e.g the labels
    classToCluster = []
    N = 0
    purity = 0
    for c in clusters:  # loop over the number of cluster
        # create an array of elements, each element is the index of the c-th element of the kmeans label
        # (Ex first step c=0), so find in kmeans_labels all the values equal to 0,
        # and saves the indices in the indices array.
        indices = [i for i in range(len(kmeansLabels)) if kmeansLabels[i] == c]

        # take all the prediction for the c-th cluster
        selectedClasses = [y[i] for i in indices]
        maxClass = 0
        maxPCF = 0
        for cl in classes:
            pcf = selectedClasses.count(cl)
            N = N + pcf
            print('cluster ', c, ' class ', cl, ' pcf ', pcf)
            if pcf > maxPCF:
                maxPCF = pcf  # predicted class frequency
                maxClass = cl
        # maxClass is the class mostly associated to the cluster c
        purity = purity + maxPCF
        classToCluster.append(maxClass)
    purity = purity / N
    return clusters, classToCluster, purity
