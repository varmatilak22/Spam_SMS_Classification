from sklearn.ensemble import RandomForestClassifier
import joblib
from data_preprocessing import load_data,preprocess
def train_model(X_train,y_train):
    """
    Function will train the model 
    And fit it on Train datset.
    """
    rf=RandomForestClassifier(n_estimators=50, max_depth=20,random_state=1)
    rf.fit(X_train,y_train)

    #Save the model
    joblib.dump(rf,'../model/randomforest.joblib')
    print("Model Saved!!!")

if __name__=='__main__':
    data=load_data()
    X_train,X_test,y_train,y_test=preprocess(data)
    train_model(X_train,y_train)