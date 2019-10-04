import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def dataPreprocessing():
    """
    Encoding of raw data
    :return:
    """

    website = []
    userInteraction = {}
    with open('test.txt', 'r') as file:
        user = 0
        interaction = []
        for line in file:
            lineContent = line.split(',')
            if lineContent[0] == 'A':
                website.append(int(lineContent[1]))
            if lineContent[0] == 'C':
                if len(interaction) > 0:
                    userInteraction[user] = interaction
                    interaction = []
                user = int(lineContent[2])
            if lineContent[0] == 'V':
                interaction.append(int(lineContent[1]))
        userInteraction[user] = interaction

    sortedWebsite = np.sort(np.asarray(website))
    userWeb = {}
    for user, item in userInteraction.items():
        temp = np.zeros(len(sortedWebsite))
        for website in item:
            index = np.where(sortedWebsite == website)
            temp[index] = 1

        userWeb[user] = temp.tolist()
    return userWeb,userInteraction


def kmeanClustering(X_train, X_val, y_train, y_val):
    """
    K-mean implementation
    :return:
    """
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X_train)
    kmeanLabels = kmeans.labels_
    indexOfclass = np.where(kmeanLabels == 0)
    clusterOne = y_train[indexOfclass]
    indexOfclass = np.where(kmeanLabels == 1)
    clusterTwo = y_train[indexOfclass]
    indexOfclass = np.where(kmeanLabels == 2)
    clusterThree = y_train[indexOfclass]
    indexOfclass = np.where(kmeanLabels == 3)
    clusterFour = y_train[indexOfclass]
    indexOfclass = np.where(kmeanLabels == 4)
    clusterFive = y_train[indexOfclass]
    y_pred = kmeans.predict([X_val[3]])
    return y_pred, clusterOne, clusterTwo, clusterThree, clusterFour, clusterFive


def hierachicalClustering(X_train, X_val, y_train, y_val):
    """
    Hierarchical implementation
    :return:
    """
    clustering = AgglomerativeClustering().fit(X_train)
    labels = clustering.labels_
    indexOfclass = np.where(labels == 0)
    clusterOne = y_train[indexOfclass]
    indexOfclass = np.where(labels == 1)
    clusterTwo = y_train[indexOfclass]
    indexOfclass = np.where(labels == 2)
    clusterThree = y_train[indexOfclass]
    indexOfclass = np.where(labels == 3)
    clusterFour = y_train[indexOfclass]
    indexOfclass = np.where(labels == 4)
    clusterFive = y_train[indexOfclass]
    y_pred = clustering.fit_predict(X_val)
    return y_pred, clusterOne, clusterTwo, clusterThree, clusterFour, clusterFive

def dbscanClustering(X_train, X_val, y_train, y_val):
    """
    DBSCAN implementation
    :return:
    """
    db = DBSCAN(eps=0.3, min_samples=10).fit(X_train)
    labels = db.labels_
    indexOfclass = np.where(labels == 0)
    clusterOne = y_train[indexOfclass]
    indexOfclass = np.where(labels == 1)
    clusterTwo = y_train[indexOfclass]
    indexOfclass = np.where(labels == 2)
    clusterThree = y_train[indexOfclass]
    indexOfclass = np.where(labels == 3)
    clusterFour = y_train[indexOfclass]
    indexOfclass = np.where(labels == 4)
    clusterFive = y_train[indexOfclass]
    y_pred = db.fit_predict(X_val)
    return y_pred, clusterOne, clusterTwo, clusterThree, clusterFour, clusterFive

def predictWebsites(userInteraction,y_pred, clusterOne, clusterTwo, clusterThree, clusterFour, clusterFive):
    """
    Prediction
    :return:
    """
    webList=[]
    count=0
    backup=[]
    if y_pred[0]==0:
        for user in clusterOne:
            if count==0:
                webList=userInteraction[user]
                backup=webList
                count=1
            else:
                webList=set(webList) & set(userInteraction[user])
            if len(webList)==0:
                webList=backup

    if y_pred[0] == 1:
        for user in clusterTwo:
            if count == 0:
                webList = userInteraction[user]
                backup = webList
                count = 1
            else:
                webList = set(webList) & set(userInteraction[user])

            if len(webList)==0:
                webList=backup

    if y_pred[0] == 2:
        for user in clusterThree:
            if count == 0:
                webList = userInteraction[user]
                backup = webList
                count = 1
            else:
                webList = set(webList) & set(userInteraction[user])

            if len(webList)==0:
                webList=backup

    if y_pred[0] == 3:
        for user in clusterFour:
            if count == 0:
                webList = userInteraction[user]
                backup = webList
                count = 1
            else:
                webList = set(webList) & set(userInteraction[user])
                if len(webList) == 0:
                    webList = backup
    if y_pred[0] == 4:
        for user in clusterFive:
            if count == 0:
                webList = userInteraction[user]
                backup = webList
                count = 1
            else:
                webList = set(webList) & set(userInteraction[user])
            if len(webList)==0:
                webList=backup
    return webList

if __name__ == '__main__':
    """
    Main function 
    
    """
    userWeb,userInteraction = dataPreprocessing()
    users = np.array(list(userWeb.keys()))
    userValues = np.array(list(userWeb.values()))
    X_train, X_val, y_train, y_val = train_test_split(userValues, users, test_size = 0.3, random_state = 42)

    y_pred, clusterOne, clusterTwo, clusterThree, clusterFour, clusterFive=kmeanClustering(X_train, X_val, y_train, y_val)
    predicteOutput= predictWebsites(userInteraction,y_pred, clusterOne, clusterTwo, clusterThree, clusterFour, clusterFive)
    print(predicteOutput)
    y_pred, clusterOne, clusterTwo, clusterThree, clusterFour, clusterFive =hierachicalClustering(X_train, X_val, y_train, y_val)
    predicteOutput = predictWebsites(userInteraction, y_pred, clusterOne, clusterTwo, clusterThree, clusterFour,
                                     clusterFive)
    print(predicteOutput)
    y_pred, clusterOne, clusterTwo, clusterThree, clusterFour, clusterFive = dbscanClustering(X_train, X_val, y_train, y_val)
    predicteOutput = predictWebsites(userInteraction, y_pred, clusterOne, clusterTwo, clusterThree, clusterFour,
                                     clusterFive)
    print(predicteOutput)

