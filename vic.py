
#Libraries used. The tensorflow libraries can be used if trying to implement the Keras models.
#The SupervisedDBNClassification library can be found at https://github.com/albertbup/deep-belief-network/blob/master/dbn/models.py

import pandas as pd
import numpy as np
import pickle
import threading
import os

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from sklearn.base import BaseEstimator, ClassifierMixin

#Additional libraries
# import tensorflow as tf
# from tensorflow import keras
# import keras.backend as K
# import concurrent.futures
# from tensorflow.python.client import device_lib
# from dbn.tensorflow import SupervisedDBNClassification

class BNHandler(BaseEstimator, ClassifierMixin):

    def __init__(self,split): 
        self.model = {}
        self.split = split

    def fit(self,X,y=None):   
        self.model = SupervisedDBNClassification()
        self.model.fit(X,y)
        #You can add more models in this part
    
    def predict(self,test_set):
        y_pred = []
        y_pred = self.model.predict(test_set)
        return y_pred

#Method that defines the number of classes for the dataset.
def createClasses(split,dataset,seed):
    rest = len(dataset)-split
        
    label0 = [0]*split
    label1 = [1]*rest
    labels = label0+label1
    
    DF = pd.DataFrame(dataset)
    DF["class"] = labels
            
    DF = shuffle(DF,random_state=seed)

    return DF


#Method that creates the splits and runs the classifiers. This method is the one called by each Thread.
def runSplits(iden,dataset,splits,seed):
    print("Running with :" + str(iden))
    print("Number of splits: " + str(len(splits)))
    clusters = {}
    best = 0
    clusters["list"] = []

    for split in splits:
        
        #Defines the label of the dataset.
        DF = createClasses(split,dataset,seed)

        x_DF = DF.iloc[:,:-1].values
        y_DF = DF.iloc[:,-1]
        
        V = []
        
        #Define the list of classifiers to use
        clfs = [SVC(),RandomForestClassifier(),GaussianNB(),LinearDiscriminantAnalysis(),MLPClassifier()]

        #Create the kfolds with 5 folds. Obtain the AUC for each classifier.
        for clf in clfs:
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            aucTotal = 0
            for train_index, test_index in kfold.split(x_DF, y_DF):
                X_train, X_test = x_DF[train_index], x_DF[test_index]
                y_train, y_test = y_DF[train_index], y_DF[test_index]
                clf.fit(X_train,y_train)
                y_pred = clf.predict(X_test)
                prec, rec, thres = precision_recall_curve(y_test, y_pred)
                aucV = auc(rec, prec)
                aucTotal+=aucV
            V.append(aucTotal/5)

        #Sort to obtain the highest AUC   
        V.sort(reverse = True)
        
        cluster = {}
        cluster["split"] = split
        cluster["x_DF"] = x_DF
        cluster["y_DF"] = y_DF
        cluster["bestAUC"] = V[0]
        cluster["allAUC"] = V
        
        clusters["list"].append(cluster)
        
        if(V[0] > best):
            clusters["best"] = cluster
            best = V[0]

    #we save the results in a pickle to be read by the main method
    filename  = "clusters"+str(iden)
    pickle.dump(clusters, open(filename, 'wb')) 
    print("Saved : ",iden)

#Define the number of threads to run for the VIC 
def VIC(dataset,splits=[],seed=30,thCount=2):
    
    g_clusters = {}
    g_clusters["list"] = list()
    g_best = 0
    threads = list()
    
    s_groups = list()
    cg = 0

    #Initialize the list of results
    for sgindx in range(thCount):
        s_groups.append(list())

    #We divide the splits into different threads
    for split in splits:
        s_groups[cg].append(split)
        cg = cg+1
        if(cg >= thCount):
            cg = 0

    #Initialize the threads
    for thindx in range(thCount):
        x = threading.Thread(target=runSplits, args=(thindx,dataset,s_groups[thindx],seed,))
        threads.append(x)
        x.start()

    #Wait for the threads to finish
    for index, thread in enumerate(threads):
        thread.join()

    #Read the results and delete the files.
    for thindx in range(thCount):
        filename  = "clusters"+str(thindx)
        loaded_model = pickle.load(open(filename, 'rb'))
        g_clusters["list"].append(loaded_model)
        mBest = loaded_model["best"]["bestAUC"]
        if(mBest > g_best):
            g_clusters["best"] = loaded_model["best"]
            g_best = mBest

        os.remove(filename)

    return g_clusters

#Method to use if trying to implement the tensorflow algorithm
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

#Method to use if trying to implement tensorflow
# gpus = get_available_gpus()
# for gpu in gpus:
#     with tf.Session(graph=tf.Graph()) as sess:
#         K.set_session(sess)
#         with tf.device(gpu):
#             clf = BNHandler(split)
#             clf.fit(x,y)
#             clf.predict(xtest)

loginL = pd.read_csv("loginL3.csv")
loginL = loginL.iloc[:,2:]

#Define the splits
splits = [100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,255,260,265,270,275,280,285,290,295,300,305,310,315,320,325,330,335,340,345]

clstrs = VIC(loginL.values,splits,30,5)

print("\n------------RESULTS------------")
print("Best split: ")
print(clstrs["best"]["split"])

print("Best AUC: ")
print(clstrs["best"]["bestAUC"])