# Import necessary functions from other modules
from data_preprocessing import load_data, preprocess
from model_training import train_model
from model_evaluation import evaluate

def run_pipeline():
    """
    This function automates the entire machine learning workflow pipeline, including:
    1. Data Extraction
    2. Data Preprocessing
    3. Model Training
    4. Model Evaluation
    5. Pipeline Integration
    6. Deployment (to be implemented as needed)
    """
    # Load the dataset using the load_data function
    data = load_data()
    
    # Preprocess the data and split it into training and testing sets
    X_train, X_test, y_train, y_test = preprocess(data)
    
    # Evaluate the model using the provided path to the trained model and the test data
    confu_mat, report = evaluate('../model/randomforest.joblib', X_test, y_test)
    
    # Print the confusion matrix
    print(f"Confusion Matrix:\n{confu_mat}")
    
    # Print the classification report
    print(f"Classification Report:\n{report}")

if __name__ == "__main__":
    # Execute the run_pipeline function if this script is run directly
    run_pipeline()
