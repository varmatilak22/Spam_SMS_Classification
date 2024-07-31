import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_data
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import os

def plot_imbalance():
    """
    It checks the imbalance in the training data
    means there are majority points for one class and minimum points for second class
    """
    data = load_data()
    X = data['text']
    y = data['class']

    # Ensure NLTK stopwords corpus is available
    nltk_data_dir = os.path.join(os.path.dirname(__file__), '../data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    nltk.data.path.append(nltk_data_dir)
    try:
        stop_words = set(nltk.corpus.stopwords.words('english'))
    except LookupError:
        print("Stopwords resource not found. Please check the directory path and structure.")

    # Convert text data into numerical features
    vectorizer = TfidfVectorizer(stop_words='english')
    X_vector = vectorizer.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_vector, y, test_size=0.2, random_state=1)

    # Plot class distribution before resampling
    plt.figure(figsize=(6,4))
    y_train_series = pd.Series(y_train)
    sample = y_train_series.value_counts()
    sns.barplot(x=sample.index, y=sample.values, palette=['blue', 'orange'])
    #plt.title('Class Distribution Before Resampling')
    plt.ylabel("Number of Instances")
    plt.savefig('before.png')

    # Apply SMOTE
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Plot class distribution after resampling
    plt.figure(figsize=(6,4))
    y_train_resampled_series = pd.Series(y_train_resampled)
    y_sample_resample = y_train_resampled_series.value_counts()
    sns.barplot(x=y_sample_resample.index, y=y_sample_resample.values, palette=['yellow'])
    #plt.title('Class Distribution After Resampling')
    plt.ylabel("Number of Instances")

    plt.savefig('after.png')


if __name__ == '__main__':
    plot_imbalance()
