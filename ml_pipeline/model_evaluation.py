import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import confusion_matrix,classification_report
import joblib
from data_preprocessing import load_data, preprocess 
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import pandas as pd

def evaluate(model_path,X_test,y_test):
    """
    returns the evaluation metrics like confusion_matrix or classification report
    """
    loaded_model=joblib.load(model_path)
    if isinstance(loaded_model,RandomForestClassifier):
        print("Model Loaded Succesfully")
    else:
        print("Model nOt Loaded")
    pred=loaded_model.predict(X_test)

    confu_matrix=confusion_matrix(y_test,pred)
    print(confu_matrix)

    report=classification_report(y_test,pred,output_dict=True)
    print(report)
    return confu_matrix,report

def visualise(confu_mat,report):
    """
    Data Visualisation using matplotlib and seaborn
    Visualisation of Confusion matrix and Classification report
    """
    plt.figure(figsize=(7,5))
    ax=sns.heatmap(confu_mat,annot=True,cmap='Greens',fmt='d')
    plt.title("Confusion Matrix:test_data")
    ax.set_xlabel('Actual Class')
    ax.set_ylabel("Predicted Class")
    ax.set_yticklabels(['ham','spam'])
    ax.set_xticklabels(['ham','spam'])
    plt.savefig('confusion_matrx.png')

    plt.figure(figsize=(8,5))
    report_df=pd.DataFrame(report).transpose()
    print(report_df)
    accuracy=report_df.loc["accuracy","precision"]
    report_df=report_df.drop(['accuracy','macro avg','weighted avg'])
    ax_1=sns.heatmap(report_df[['precision','recall','f1-score']],annot=True,cmap='Greens',fmt='.2f')
    ax_1.set_xlabel("Metrics")
    ax_1.set_ylabel("Classes")
    ax_1.set_yticklabels(['ham','spam'])
    ax_1.set_xticklabels(['precision','recall','f1-score'])
    ax_1.set_title(f'Classification Report:{accuracy:.2f}')
    plt.savefig('report.png')
    #print(report_df.loc['accuracy'])





if __name__=='__main__':
    data=load_data()
    X_train,X_test,y_train,y_test=preprocess(data)
    confu_mat,report=evaluate('../model/randomforest.joblib',X_test,y_test)
    visualise(confu_mat,report)


