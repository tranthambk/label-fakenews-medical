import re
import pandas as pd
import re
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from pyvi import ViTokenizer
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
import pickle
import json
import re
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score
import datetime                            # Imports datetime library

from pymongo import MongoClient
from collections import Counter
from src.util import config as cf



def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F300-\U0001FAD6"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)
    
def preprocess(texts):
    texts = [i.lower() for i in texts]
    texts = [remove_emoji(i) for i in texts]

    texts = [re.sub('[^\w\d\s]', '', i) for i in texts]
    
    texts = [re.sub('\s+|\n', ' ', i) for i in texts]
    texts = [re.sub('^\s|\s$', '', i) for i in texts]

    texts = [ViTokenizer.tokenize(i) for i in texts]

    return texts
def get_client():
    # uri (uniform resource identifier) defines the connection parameters 
    uri = cf.URI_DB
    # start client to connect to MongoDB server 
    client = MongoClient( uri )
    return client


def detect_fakenews():


    client = get_client()

    # Show existing database names
    name_db = cf.NAME_DB
    name_collection = cf.NAME_COLLECTION

    db = client[name_db][name_collection]
    df_not_predict =  pd.DataFrame(list(db.find({"is_medical": {"$eq": True}, "is_fakenew": {"$exists": 0}})))
    df_not_predict.shape
    if df_not_predict.shape[0] > 0:
        df = pd.DataFrame(list(db.find({"is_medical": {"$eq": True}, "is_fakenew": {"$exists": 1}})))
        print(df.shape)
        print(df.head(5))
        df["label"] = df.is_fakenew.apply(lambda x: 1 if x == True else 0 if x == False else -1 )
        print(df.label.value_counts())
        print(df.is_fakenew.value_counts())
        df = df[df.label.isin([0,1])]
        df = df[df.text != ""]
        texts = preprocess(list(df.text.values))
        print(texts[:2])
        labels = df.label.values

        # TFIDF
        tf_vectorizer=TfidfVectorizer(max_features=3000)
        texts_tfidf=tf_vectorizer.fit_transform(texts) 
        texts_tfidf
        # texts_tfidf=tf_vectorizer.transform(texts)
        x_train,x_val,y_train,y_val=train_test_split(texts_tfidf,labels,test_size=0.15)
        print(x_train.shape, np.array(y_train).shape, x_val.shape, np.array(y_val).shape)

        clf = SVC(kernel='rbf', gamma=1e-1, C=4)
        clf.fit(x_train, y_train)
        clf.score(x_val, y_val)
        y_predict = clf.predict(x_val)
        cf_matrix = confusion_matrix(y_val,y_predict)
        precision, recall, fscore, support = score(y_val, y_predict)

        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))
        df_cm = pd.DataFrame(cf_matrix, index = [i for i in "01"],
                        columns = [i for i in "01"])
        print(df_cm)
        
        
        #update 


        df_not_predict = df_not_predict[df_not_predict.text != ""]
        print(df_not_predict.shape)
        texts_new = preprocess(list(df_not_predict.text.values))
        texts_new_tfidf = tf_vectorizer.transform(texts_new)
        print(texts_new_tfidf[:2])
        y_new_pred = clf.predict(texts_new_tfidf)
        print(Counter(y_new_pred))
        df_not_predict["y_pred"] = y_new_pred
        print(df_not_predict["y_pred"].value_counts())
        print(df_not_predict.post_id.values)
        print(df_not_predict.y_pred.values)
        #fake new
        ls_id = list(df_not_predict[df_not_predict.y_pred == 1].post_id.astype(str).values)
        print(len(ls_id))
        # print(type(ls_id[0]))
        if len(ls_id) > 0:

            print("NUMBER DOCUMENT: ", db.count_documents({"is_medical": {"$eq": True}, "is_fakenew": {"$exists": 0}, "post_id": {"$in": ls_id} }))
            db.update_many({"is_medical": {"$eq": True}, "is_fakenew": {"$exists": 0}, "post_id": {"$in": ls_id} }, {"$set": {"is_fakenew": True, "is_auto_fakenew": True, "is_verify_fakenew": False } })
            # df_new = pd.DataFrame(db.find({"is_medical": {"$exists": False}, "post_id": {"$in": ls_id} }))
            # df_new
        ls_id = list(df_not_predict[df_not_predict.y_pred == 0].post_id.astype(str).values)
        if len(ls_id) > 0:

            print("NUMBER DOCUMENT: ", db.count_documents({"is_medical": {"$eq": True}, "is_fakenew": {"$exists": 0}, "post_id": {"$in": ls_id} }))
            db.update_many({"is_medical": {"$eq": True}, "is_fakenew": {"$exists": 0}, "post_id": {"$in": ls_id} }, {"$set": {"is_fakenew": False, "is_auto_fakenew": True, "is_verify_fakenew": False } })




def detect_medical():


    client = get_client()

    # Show existing database names
    name_db = cf.NAME_DB
    name_collection = cf.NAME_COLLECTION

    db = client[name_db][name_collection]
    df_not_predict =  pd.DataFrame(list(db.find({"is_medical": {"$exists": 0}})))
    if df_not_predict.shape[0] > 0:
        df = pd.DataFrame(list(db.find({"post_id": {"$exists": True}})))
        print(df.shape)
        print(df)

        df["label"] = df.is_medical.apply(lambda x: 1 if x == True else 0 if x == False or x == "false" else -1 )
        print(df.label.value_counts())
        print(df.is_medical.value_counts())
        df = df[df.label.isin([0,1])]
        df = df[df.text != ""]
        texts = preprocess(list(df.text.values))
        print(texts[:2])
        labels = df.label.values

        # TFIDF
        tf_vectorizer=TfidfVectorizer(max_features=3000)
        texts_tfidf=tf_vectorizer.fit_transform(texts) 
        texts_tfidf
        # texts_tfidf=tf_vectorizer.transform(texts)


        x_train,x_val,y_train,y_val=train_test_split(texts_tfidf,labels,test_size=0.15)
        print(x_train.shape)
        print(np.array(y_train).shape)
        print(x_val.shape)
        print(np.array(y_val).shape)

        clf = SVC(kernel='rbf', gamma=1e-1, C=4)
        clf.fit(x_train, y_train)
        clf.score(x_val, y_val)



        y_predict = clf.predict(x_val)
        cf_matrix = confusion_matrix(y_val,y_predict)
        precision, recall, fscore, support = score(y_val, y_predict)

        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))
        df_cm = pd.DataFrame(cf_matrix, index = [i for i in "01"],
                        columns = [i for i in "01"])


        #update 

    
        df_not_predict = df_not_predict[df_not_predict.text != ""]
        print(df_not_predict.shape)
        texts_new = preprocess(list(df_not_predict.text.values))
        texts_new_tfidf = tf_vectorizer.transform(texts_new)
        print(texts_new_tfidf[:2])
        y_new_pred = clf.predict(texts_new_tfidf)
        print(Counter(y_new_pred))
        df_not_predict["y_pred"] = y_new_pred
        print(df_not_predict["y_pred"].value_counts())
        #fake new
        ls_id = list(df_not_predict[df_not_predict.y_pred == 1].post_id.astype(str).values)
        print(len(ls_id))
        # print(type(ls_id[0]))

        print("NUM DOCUMENT: ", db.count_documents({"is_medical": {"$exists": False}, "post_id": {"$in": ls_id} }))
        """
    #     db.update_many({"is_medical": {"$exists": False}, "post_id": {"$in": ls_id} }, {"$set": {"is_medical": True, "is_auto": True, "is_verify": False } })
    #     # df_new = pd.DataFrame(db.find({"is_medical": {"$exists": False}, "post_id": {"$in": ls_id} }))
    #     # df_new
        """

        ls_id = list(df_not_predict[df_not_predict.y_pred == 1].post_id.astype(str).values)
        print(len(ls_id))
        print("LS ID: ", ls_id)
        # print(type(ls_id[0]))
        if len(ls_id) > 0:

            print("NUMBER DOCUMENT: ", db.count_documents({"is_medical": {"$eq": True}, "post_id": {"$in": ls_id} }))
            db.update_many({"is_medical": {"$exists": 0}, "post_id": {"$in": ls_id} }, {"$set": {"is_medical": True, "is_auto_medical": True, "is_verify_medical": False } })
            # df_new = pd.DataFrame(db.find({"is_medical": {"$exists": False}, "post_id": {"$in": ls_id} }))
            # df_new
        ls_id = list(df_not_predict[df_not_predict.y_pred == 0].post_id.astype(str).values)
        if len(ls_id) > 0:

            print("NUMBER DOCUMENT: ", db.count_documents({"is_medical": {"$eq": True}, "is_fakenew": {"$exists": 0}, "post_id": {"$in": ls_id} }))
            db.update_many({"is_medical": {"$eq": True}, "is_fakenew": {"$exists": 0}, "post_id": {"$in": ls_id} }, {"$set": {"is_medical": False, "is_auto_medical": True, "is_verify_medical": False } })











