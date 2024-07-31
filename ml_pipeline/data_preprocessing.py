import pandas as pd
import numpy as np
import psycopg2
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib 

#Load data
def load_data():
    """
    retrives the data from the postgres database.
    return the data
    """
    connection=psycopg2.connect(
        dbname='sampledb',
        port='5432',
        user='postgres',
        host='localhost',
        password='1234'
    )

    cursor=connection.cursor()
    cursor.execute("select * from spam;")
    data=cursor.fetchall()

    #Convert the data into pandas dataframe
    data=pd.DataFrame(data)
    
    #Drop empy columns 
    data=data.drop([2,3,4],axis=1)
    
    #Rename columns names
    col_names={0:'class',1:'text'}
    data.rename(columns=col_names,inplace=True)
    
    return data

def preprocess(data):
    """
    We have to preprocess the text data:
    Text preprocessing are:
    1. Lowercase
    2. Removing HTML tags,special characters
    3. Stopwords removal
    4. Stemming/lemmatization
    5. Tokenisation
    6. TextVectorization/CountVectorization

    Here in this text preprocessing TF-IDf
    Term frequency and inverse document frequency 
    """
    #Shuffle the data
    data=data.sample(frac=1,random_state=1).reset_index(drop=True)

    #Separate features and target 
    X_text=data['text']
    y=data['class']

    #Convert the text into numercial features
    from nltk.corpus import stopwords
    # Path where the stopwords corpus is located
    nltk_data_dir = os.path.join(os.path.dirname(__file__), '../data')

    # Ensure the directory exists
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)

    # Update NLTK data path
    nltk.data.path.append(nltk_data_dir)
    count=0
    try:
       # Test if stopwords can be accessed
       stop_words = set(stopwords.words('english'))

    except LookupError:
       print("Stopwords resource not found. Please check the directory path and structure.")

    vectorizer=TfidfVectorizer(stop_words='english', max_df=0.75, min_df=2)
    X=vectorizer.fit_transform(X_text)
    print(X)
    joblib.dump(vectorizer,'../model/vectorizer.pkl')

    #Split the data into training and test sets
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

    #Apply SMOTE
    smote=SMOTE()
    X_train_resampled,y_train_resampled=smote.fit_resample(X_train,y_train)

    return X_train_resampled,X_test,y_train_resampled,y_test
    

if __name__=='__main__':
    data=load_data()
    print(data)
    X_train,X_test,y_train,y_test=preprocess(data)
    




