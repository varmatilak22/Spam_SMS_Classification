from data_preprocessing import load_data,preprocess
from model_training import train_model
from model_evaluation import evaluate 

def run_pipeline():
    """
    This is the pipeline where we we automate our workflow 
    steps:
    1. Data Extracttion
    2. Data Preproessing
    3. Model Training 
    4. MOdel Evaluation
    5. Pipeline
    6. Deployment
    """
    data=load_data()
    X_train,X_test,y_train,y_test=preprocess(data)
    confu_mat,report=evaluate('../model/randomforest.joblib',X_test,y_test)
    print(f"Confusion Matrix:{confu_mat}")
    print(f"Classification Report")
    print(report)


if __name__=="__main__":
    run_pipeline()