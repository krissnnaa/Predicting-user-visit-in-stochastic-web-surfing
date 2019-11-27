import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
import operator
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from yellowbrick.cluster import KElbowVisualizer

def dataPreprocessing(filename):
    """
    Encoding of raw data
    :return:
    """
    website = []
    userInteraction = {}
    with open(filename, 'r') as file:
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
    if filename.startswith('train')==True:
        for user, item in userInteraction.items():
            temp = np.zeros(len(sortedWebsite))
            for website in item:
                index = np.where(sortedWebsite == website)
                temp[index] = 1
            userWeb[user] = temp.tolist()
        return userWeb, userInteraction
    else:
        testLabel={}
        for user, item in userInteraction.items():
            temp = np.zeros(len(sortedWebsite))
            if len(item)==1:
                testLabel[user]=item[0]
                index = np.where(sortedWebsite == item[0])
                temp[index] = 1
            else:
                testLabel[user] = item[len(item)-1]
                for website in item[:len(item)-1]:
                    index = np.where(sortedWebsite == website)
                    temp[index] = 1
            userWeb[user] = temp.tolist()
        return userWeb,testLabel




def userBasedNeighboring(X_train,userInteract,X_test, testUser):
    #Training
    cosine=cosine_similarity(X_train)
    indx=np.argsort(cosine,axis=1)
    similarTen={}
    for user,simUser in zip(userInteract.keys(),indx):
        similarTen[user]=list(simUser[len(X_train)-10:])
    predictedWeb={}
    for simUs,simVal in similarTen.items():
        temp=[]
        for item in simVal:
            item=(10001+int(item))
            temp.append(userInteract[item])
        flat_list = [item for sublist in temp for item in sublist]

        count=dict(Counter(flat_list))
        clusteList = sorted(count.items(), key=lambda kv: kv[1])
        finalList = []
        for web, count in clusteList[len(clusteList) - 10:]:
            finalList.append(web)
        predictedWeb[simUs]= finalList #max(count.items(), key=operator.itemgetter(1))[0]

    #Testing
    testCosine=cosine_similarity(X_test,X_train)
    indx = np.argsort(testCosine, axis=1)
    finalIndex=indx[:,-1:]
    truePredict=0
    i=10001
    for ind in finalIndex:
        for web in predictedWeb[int(ind)+10001]:
            if int (web)==int(testUser[i]):
                truePredict+=1
        i+=1

    testAccuracy = truePredict / 5000

    print('Test accuracy using user-based neighboring method=')
    print(testAccuracy)


def kmeanClustering(X_train,userInteract,X_test, testUser):
    """
    K-mean implementation
    :return:
    """
    #Training
    kmeans = KMeans(n_clusters=7, random_state=0).fit(X_train)
    kmeanLabels = kmeans.labels_
    predictDict={}
    indexOfclass = np.asarray(np.where(kmeanLabels == 0))
    predictDict[0] = predictWebsites(userInteract,indexOfclass+10001)
    indexOfclass = np.asarray(np.where(kmeanLabels == 1))
    predictDict[1] = predictWebsites(userInteract, indexOfclass+10001)
    indexOfclass = np.asarray(np.where(kmeanLabels == 2))
    predictDict[2] = predictWebsites(userInteract, indexOfclass+10001)
    indexOfclass = np.asarray(np.where(kmeanLabels == 3))
    predictDict[3] = predictWebsites(userInteract, indexOfclass+10001)
    indexOfclass = np.asarray(np.where(kmeanLabels == 4))
    predictDict[4] = predictWebsites(userInteract, indexOfclass+10001)
    indexOfclass = np.asarray(np.where(kmeanLabels == 5))
    predictDict[5] = predictWebsites(userInteract, indexOfclass+10001)
    indexOfclass = np.asarray(np.where(kmeanLabels == 6))
    predictDict[6]= predictWebsites(userInteract, indexOfclass+10001)
    centroids=kmeans.cluster_centers_

    # Testing
    testCosine = cosine_similarity(X_test, centroids)
    indx = np.argsort(testCosine, axis=1)
    finalIndex = indx[:, -1:]
    truPredict=0

    for clustNum,target in zip(finalIndex,testUser.values()):
        for web in predictDict[int(clustNum)]:
                if int(web)==int(target):
                    truPredict+=1

    testAccuracy=truPredict/5000

    print('Test accuracy=')
    print(testAccuracy)


    elbowPlot(kmeans,X_train)


def elbowPlot(model,data):
    visualizer = KElbowVisualizer(model, k=(1, 12))
    visualizer.fit(data)
    visualizer.show()




def predictWebsites(userInteraction,clusterWeb):
    """
    Prediction
    :return:
    """
    webList=[]

    for user in clusterWeb[0]:
        webList.append(userInteraction[user])
    flat_list = [item for sublist in webList for item in sublist]
    count = dict(Counter(flat_list))
    clusteList=sorted(count.items(), key=lambda kv: kv[1])
    finalList=[]
    for web,count in clusteList[len(clusteList)-3:]:
        finalList.append(web)
    # singleMax=max(count.items(), key=operator.itemgetter(1))[0]
    return (finalList)



if __name__ == '__main__':
    """
    Main function 
    
    """
    userWeb,userInteraction = dataPreprocessing('training.txt')
    users = np.array(list(userWeb.keys()))
    userValues = np.array(list(userWeb.values()))
    testuserWeb,testLabel=dataPreprocessing('test.txt')
    testusers = np.array(list(testuserWeb.keys()))
    testuserValues = np.array(list(testuserWeb.values()))

    # User-based neighboring
    userBasedNeighboring(userValues,userInteraction,testuserValues,testLabel)

    # k-mean clustering
    kmeanClustering(userValues,userInteraction,testuserValues,testLabel)
