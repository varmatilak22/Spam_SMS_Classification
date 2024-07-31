# Spam SMS Classification Project 📱🕵️‍♂️

## Overview
This project aims to classify SMS messages as spam or non-spam using natural language processing (NLP) and machine learning techniques. We use the `TfidfVectorizer` for text preprocessing and various machine learning algorithms to identify and categorize SMS content effectively.

Project link [Spam_SMS_Classification](https://spam-sms-classification.streamlit.app/)

## Table of Contents
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Data](#data)
- [Text Preprocessing](#text-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Class Imbalance & SMOTE](#class-imbalance--smote)
- [Optimization](#optimization)
  - [Grid Search](#grid-search)
  - [Random Search](#random-search)
- [Real World Examples](#real-world-examples)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Getting Started
Follow these instructions to set up and run the project locally for development and testing purposes.

## Prerequisites
Ensure that you have Python installed along with the required libraries. Download Python from [python.org](https://www.python.org/) and install libraries using pip.

## Data
The dataset for this project is located in the data directory. The primary file is `sms_spam.csv`, which contains SMS messages labeled as spam or ham (non-spam). The dataset is sourced from Kaggle.

## Text Preprocessing
We use the `TfidfVectorizer` to preprocess SMS messages, transforming raw text into numerical features that machine learning algorithms can use.

1. **TF-IDF Vectorization**:
   - **Term Frequency-Inverse Document Frequency (TF-IDF)** is a technique that converts text into numerical features based on the importance of each word in the document relative to the entire corpus.
   - **Term Frequency (TF)** measures how often a word appears in a document.
   - **Inverse Document Frequency (IDF)** adjusts the term frequency by how often the word appears across all documents. It reduces the weight of common words and increases the weight of rare words.
   - **TF-IDF Score** is the product of TF and IDF, representing the importance of a word in a document.

   Example:
   - For the message "Win a $100 gift card", TF-IDF will assign higher scores to less common words like "gift" and "card" compared to common words like "win" and "a".

## Model Training
The model training process is detailed in the `model_train.py` script. We utilize decision tree and random forest algorithms to train models on the preprocessed SMS data.

**Decision Tree**:
   - A popular classification model that splits the data into smaller regions based on feature values, making predictions based on these splits.
   - Decision trees are easy to interpret and visualize, making them useful for understanding the model's decision-making process.

**Random Forest**:
   - An ensemble learning method that combines multiple decision trees to improve accuracy and robustness.
   - Random forests aggregate predictions from multiple trees to provide a final classification decision.

## Evaluation
The performance of the models is evaluated using metrics such as accuracy, precision, recall, and F1-score. Detailed results are available in the `model_evaluation.py` script.

## Class Imbalance & SMOTE
Spam SMS classification often faces class imbalance, where one class (e.g., spam) may be underrepresented compared to the other (e.g., ham). To address this:

1. **Class Imbalance**:
   - Class imbalance can lead to biased models that perform poorly on the minority class.
   - Evaluating class distribution helps in understanding the extent of imbalance.

2. **SMOTE (Synthetic Minority Over-sampling Technique)**:
   - SMOTE generates synthetic samples to balance the class distribution.
   - By creating new instances of the minority class, SMOTE helps the model learn better and improve classification performance.

**Visualization**:
   - **Class Distribution Before SMOTE**: Visualize the imbalance in the dataset.
   - **Class Distribution After SMOTE**: Show the balanced distribution post-SMOTE application.

## Optimization
We optimize our models using hyperparameter tuning techniques to enhance performance.

### Grid Search
- **Grid Search** exhaustively searches through a specified range of hyperparameters to find the best combination.
- It evaluates all possible combinations within the defined range and selects the optimal set based on performance metrics.

### Random Search
- **Random Search** samples random combinations of hyperparameters from a defined range.
- It is often faster than Grid Search and can efficiently find good hyperparameters without exhaustive searching.

**Visualization**:
   - **Grid Search Optimization Results**: Display the best parameters found through Grid Search.
   - **Random Search Optimization Results**: Show the best parameters found through Random Search.

## Real World Examples
Here we showcase examples of spam and non-spam SMS messages, demonstrating the model's performance in real-world scenarios.

1. **Example 1**: "Congratulations! You've won a $1000 gift card. Call now to claim your prize!"
2. **Example 2**: "Hey, how are you? Let's catch up this weekend."
3. **Example 3**: "Your account has been compromised. Click this link to secure it immediately."
4. **Example 4**: "Are you available for a quick meeting tomorrow?"
5. **Example 5**: "Limited time offer! Buy one get one free on all items."

**Visualization**:
   - **Classification Report**: Display the classification report, showing metrics such as precision, recall, and F1-score for the real-world examples.

## Results
The trained models are saved in the models directory. You can use these models to classify new SMS messages.

## Contributing
We welcome contributions to this project! Please open an issue or create a pull request if you'd like to contribute.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
Kaggle
