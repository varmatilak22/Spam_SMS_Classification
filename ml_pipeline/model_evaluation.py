import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib
from data_preprocessing import load_data, preprocess 
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def evaluate(model_path, X_test, y_test):
    """
    Evaluates the performance of a model using confusion matrix and classification report.

    Args:
        model_path (str): Path to the trained model file.
        X_test (pd.DataFrame): Test feature data.
        y_test (pd.Series): True labels for the test data.

    Returns:
        tuple: Confusion matrix and classification report as dictionaries.
    """
    # Load the trained model from the specified path
    loaded_model = joblib.load(model_path)
    
    # Check if the loaded model is a RandomForestClassifier
    if isinstance(loaded_model, RandomForestClassifier):
        print("Model Loaded Successfully")
    else:
        print("Model Not Loaded")
    
    # Make predictions using the loaded model
    pred = loaded_model.predict(X_test)

    # Compute the confusion matrix
    confu_matrix = confusion_matrix(y_test, pred)
    print(confu_matrix)

    # Generate the classification report
    report = classification_report(y_test, pred, output_dict=True)
    print(report)
    
    return confu_matrix, report

def visualise(confu_mat, report):
    """
    Visualizes the confusion matrix and classification report using matplotlib and seaborn.

    Args:
        confu_mat (np.ndarray): Confusion matrix.
        report (dict): Classification report as a dictionary.
    """
    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(7, 5))
    ax = sns.heatmap(confu_mat, annot=True, cmap='Greens', fmt='d')
    plt.title("Confusion Matrix: Test Data")
    ax.set_xlabel('Actual Class')
    ax.set_ylabel('Predicted Class')
    ax.set_yticklabels(['ham', 'spam'])
    ax.set_xticklabels(['ham', 'spam'])
    plt.savefig('confusion_matrix.png')

    # Create a heatmap for the classification report
    plt.figure(figsize=(8, 5))
    report_df = pd.DataFrame(report).transpose()
    print(report_df)
    
    # Extract the accuracy from the classification report
    accuracy = report_df.loc["accuracy", "precision"]
    
    # Drop rows that are not related to individual classes
    report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'])
    
    # Plot the classification report
    ax_1 = sns.heatmap(report_df[['precision', 'recall', 'f1-score']], annot=True, cmap='Greens', fmt='.2f')
    ax_1.set_xlabel("Metrics")
    ax_1.set_ylabel("Classes")
    ax_1.set_yticklabels(['ham', 'spam'])
    ax_1.set_xticklabels(['precision', 'recall', 'f1-score'])
    ax_1.set_title(f'Classification Report: {accuracy:.2f}')
    plt.savefig('report.png')

if __name__ == '__main__':
    # Load and preprocess the data
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess(data)
    
    # Evaluate the model and visualize the results
    confu_mat, report = evaluate('../model/randomforest.joblib', X_test, y_test)
    visualise(confu_mat, report)
