# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
import joblib
from data_preprocessing import load_data, preprocess

def train_model(X_train, y_train):
    """
    Trains a RandomForestClassifier model and saves it to a file.

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training labels.
    """
    # Initialize the RandomForestClassifier with specified hyperparameters
    rf = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=1)
    # Fit the model on the training data
    rf.fit(X_train, y_train)
    
    # Save the trained model to a file
    joblib.dump(rf, '../model/randomforest.joblib')
    print("Model Saved!!!")

if __name__ == '__main__':
    # Load the dataset using the load_data function
    data = load_data()
    
    # Preprocess the data and split it into training and testing sets
    X_train, X_test, y_train, y_test = preprocess(data)
    
    # Train the model using the training data
    train_model(X_train, y_train)
