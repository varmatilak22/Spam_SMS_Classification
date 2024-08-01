import streamlit as st
import joblib 
import numpy as np
import os 
from helper import abs_model_path,abs_image_path
import matplotlib.pyplot as plt
from PIL import Image
def load_model_and_vectorizer():
    """
    This function will load th vectorizer and model
    from there absolute path
    """
    file_path,vector_path=abs_model_path()
    model=joblib.load(file_path)
    vector=joblib.load(vector_path)
    return model,vector

model,vector=load_model_and_vectorizer()

def predict(text):
    vectorized_text=vector.transform([text])
    predict=model.predict(vectorized_text)
    return predict



# Streamlit app
st.title('📊 Spam SMS Classification Dashboard')
st.write("Welcome to the Spam SMS Classification Dashboard. This app allows you to interact with the model, understand its workings, and evaluate its performance.")

# Navigation sidebar
st.sidebar.title('📑 Navigation')
page = st.sidebar.radio('Select a Page:', ['🔮 Predictions', '🔍 Model Explanation', '📉 Class Imbalance & SMOTE', '📈 Evaluation', '⚙️ Optimization','🌍 Real World Examples'])

if page == '🔮 Predictions':
    st.header('🔮 Predictions Section')
    st.write('Enter an SMS message below to get predictions from the trained spam classifier.')
    
    input_text = st.text_area('Enter SMS for prediction:')
    
    if st.button('🔍 Predict'):
        prediction = predict(input_text)
        if prediction is not None:
            st.write(f"Predictions:{prediction[0]}")

        #st.write('**Prediction result will be shown here**')
        # Here you would normally call a function to get prediction from the model
        # Example:
        # prediction = predict_sms(input_text)
        # st.write(f'The SMS is classified as: {prediction}')

elif page == '🔍 Model Explanation':
    st.header('🔍 Model Explanation')
    st.write('Let’s explore the components of the spam SMS classification model in detail:')
    
    # Interactive TF-IDF explanation
    st.subheader('📊 TF-IDF Vectorizer')
    
    # Interactive slider for TF-IDF parameters
    term_freq = st.slider('Term Frequency (TF):', min_value=0, max_value=100, value=50)
    inverse_doc_freq = st.slider('Inverse Document Frequency (IDF):', min_value=0, max_value=100, value=50)
    
    st.write('**How TF-IDF Works:**')
    st.write(f'- **Term Frequency (TF):** Measures how frequently a term occurs in a document. Current value: {term_freq}.')
    st.write(f'- **Inverse Document Frequency (IDF):** Measures how important a term is. Current value: {inverse_doc_freq}.')
    
    st.write('**TF-IDF Score:** Combines TF and IDF to give a score reflecting a term’s importance in the document.')
    
    # Plotting TF-IDF Example
    st.write('**Interactive TF-IDF Visualization:**')
    fig, ax = plt.subplots()
    terms = ['term1', 'term2', 'term3', 'term4']
    tfidf_scores = np.random.rand(4) * term_freq * inverse_doc_freq / 100
    
    ax.bar(terms, tfidf_scores, color='skyblue')
    ax.set_xlabel('Terms')
    ax.set_ylabel('TF-IDF Score')
    ax.set_title('TF-IDF Scores for Different Terms')
    
    st.pyplot(fig)
    # Interactive Random Forest explanation
    st.subheader('🌳 Random Forest Classifier')
    st.write('**What is Random Forest?**')
    st.write('Random Forest is like having a forest of many decision trees. Each tree makes its own decision, and the final prediction is based on the majority vote from all trees.')
    
    st.write('**How It Works:**')
    st.write('- **Decision Trees:** Each tree makes a decision based on the input data.')
    st.write('- **Forest of Trees:** Multiple trees are used to make predictions.')
    st.write('- **Majority Vote:** The final prediction is determined by the majority vote from all the trees.')
    
    st.write('**Why Use Random Forest?**')
    st.write('- **Accuracy:** Combining many trees improves prediction accuracy.')
    st.write('- **Robustness:** It handles noisy data well and reduces overfitting.')
    
    st.write('**In Simple Terms:**')
    st.write('Imagine asking many experts for their opinion. Random Forest uses the opinions of many trees to make a more reliable prediction.')

elif page == '📉 Class Imbalance & SMOTE':
    
    # Streamlit app section for Class Imbalance & SMOTE
    st.header('📉 Class Imbalance & SMOTE')
    st.write('Visualize the class distribution before and after applying SMOTE to balance the classes.')

    # Adding interactive explanations
    st.subheader('What is Class Imbalance?')
    st.write('Class imbalance happens when one class is much more frequent than another in your data. For example, in spam SMS classification, if we have many "non-spam" messages and few "spam" messages, the data is imbalanced.')

    st.write('This imbalance can lead to models that are biased towards the majority class, making them less effective at predicting the minority class.')

    st.subheader('What is SMOTE?')
    st.write('SMOTE stands for Synthetic Minority Over-sampling Technique. It is used to balance the classes by creating synthetic examples for the minority class.')

    st.write('**How SMOTE Works:**')
    st.write('1. **Identify Minority Class:** Focuses on the underrepresented class (e.g., spam messages).')
    st.write('2. **Generate Synthetic Samples:** Creates new examples by blending existing examples of the minority class.')
    st.write('3. **Balance the Dataset:** Adds these synthetic samples to balance the dataset, helping the model learn better from the minority class.')

    st.write('**Why Use SMOTE?**')
    st.write('SMOTE helps in:')
    st.write('- Improving model performance by providing more balanced data.')
    st.write('- Preventing bias towards the majority class.')
    st.write('- Enhancing the learning of patterns for the minority class.')

    # Adding images
    st.write('**Class Distribution Before SMOTE:**')
    img_path=abs_image_path('before.png')
    img=Image.open(img_path)
    st.image(img, caption='Class Distribution Before SMOTE')

    st.write('**Class Distribution After SMOTE:**')
    img_path=abs_image_path('after.png')
    img=Image.open(img_path)
    st.image(img, caption='Class Distribution After SMOTE')

elif page == '📈 Evaluation':
    st.title('📈 Model Evaluation')
    st.write('Evaluate the performance of the spam classification model using various metrics.')

    # Confusion Matrix Explanation
    st.header('🔍 Confusion Matrix')
    st.write("""
    The confusion matrix is a summary table used to evaluate the performance of a classification model. It provides insight into the number of correct and incorrect predictions made by the model.
    """)

    st.write('**Confusion Matrix Parameters:**')
    st.write('- **True Positives (TP):** Number of correctly predicted positive instances.')
    st.write('- **True Negatives (TN):** Number of correctly predicted negative instances.')
    st.write('- **False Positives (FP):** Number of incorrectly predicted positive instances (actual negative).')
    st.write('- **False Negatives (FN):** Number of incorrectly predicted negative instances (actual positive).')

    st.write('**Confusion Matrix Formula:**')
    st.latex(r'''
    \text{Confusion Matrix} =
\begin{bmatrix}
TP & FP \\
FN & TN
\end{bmatrix}
''')

    st.write('**Visualization of the Confusion Matrix:**')
    img_path=abs_image_path('confusion_matrx.png')
    img=Image.open(img_path)
    st.image(img, caption='Confusion Matrix Heatmap')

    # Classification Report Explanation
    st.header('🔍 Classification Report')
    st.write("""
    The classification report provides a detailed analysis of the classification performance using several metrics. These metrics include precision, recall, F1-score, and support.
    """)

    st.write('**Classification Report Metrics:**')
    st.write('- **Precision:** The proportion of positive identifications that were actually correct.')
    st.latex(r'''
    \text{Precision} = \frac{TP}{TP + FP}
    ''')

    st.write('- **Recall (Sensitivity or True Positive Rate):** The proportion of actual positives that were correctly identified.')
    st.latex(r'''
    \text{Recall} = \frac{TP}{TP + FN}
    ''')

    st.write('- **F1-Score:** The harmonic mean of precision and recall.')
    st.latex(r'''
    \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
    ''')

    st.write('- **Support:** The number of actual occurrences of the class in the dataset.')

    st.write('**Visualization of the Classification Report:**')
    img_path=abs_image_path('report.png')
    img=Image.open(img_path)
    st.image(img, caption='Classification Report Metrics')

elif page == '⚙️ Optimization':

    # Streamlit app
    st.title('⚙️ Model Optimization')
    st.write('Optimize the model using Grid Search and Random Search to find the best hyperparameters.')

    # Grid Search Optimization
    st.subheader('🔍 Grid Search Optimization')
    st.write("""
    Grid Search is an optimization technique where we specify a range of values for different hyperparameters. 
    It e    valuates all possible combinations within this range to find the optimal parameters for our model.
    """)     
    st.write("""
    **How It Works:**

    1. **Define Hyperparameters:** Specify the hyperparameters and their respective ranges.
    2. **Exhaustive Search:** Grid Search tries every combination of the defined hyperparameters.
    3. **Performance Evaluation:** Each combination is evaluated, and the best-performing set is selected based on a performance metric.

    **Example:**

    If we have two hyperparameters, `max_depth` and `n_estimators`, with ranges of [5, 10, 15] and [50, 100, 150] respectively, Grid Search will evaluate every combination (e.g., (5, 50), (5, 100), ..., (15, 150)) to find the best set.
    """)

    # Random Search Optimization
    st.subheader('🔍 Random Search Optimization')
    st.write("""
    Random Search randomly samples different combinations of hyperparameters from a specified range.
    This method is often faster than Grid Search and can find good hyperparameters without exhaustively searching the entire space.
    """)
    st.write("""
    **How It Works:**

    1. **Define Hyperparameters:** Specify the hyperparameters and their ranges.
    2. **Random Sampling:** Randomly select combinations of hyperparameters within the defined ranges.
    3. **Performance Evaluation:** Evaluate the model with each sampled set and choose the best-performing set.

    **Example:**

    Instead of evaluating every possible combination, Random Search might randomly choose (10, 100), (5, 150), etc., and find the best-performing set from these samples. This approach can be more efficient, especially with a large number of hyperparameters.

    **Comparison:**

    - **Grid Search:** Thorough but potentially time-consuming.
    - **Random Search:** Faster and can be more efficient with a large hyperparameter space.
    """)

    # Display optimization results (example results; replace with actual results)
    st.write('**Best parameters from Random Search:**')
    st.write("""
    - `tfidf__max_df`: 0.75
    - `tfidf__min_df`: 2
    - `clf__n_estimators`: 124
    - `clf__max_depth`: 28
    """)
    st.write('**Best score:** 0.97')


elif page == '🌍 Real World Examples':
    st.header('🌍 Real World Examples')
    st.write("""
    In this section, we present a variety of real-world SMS messages to illustrate examples of spam and non-spam content.

    **Spam SMS Examples:**
    - **Example 1:** 
      ```
      Congratulations! You've won a $1000 gift card. Reply YES to claim your prize now!
      ```
      *This message is a classic example of spam, offering an unsolicited prize and urging immediate action.*

    - **Example 2:** 
      ```
      Urgent: Your account has been compromised. Call 1-800-123-4567 to secure your account.
      ```
      *This message tries to create a sense of urgency and includes a phone number, often a tactic used in phishing scams.*

    - **Example 3:** 
      ```
      Get a free vacation package! Click the link to see if you qualify: [link]
      ```
      *This is a spam message promoting an offer that may require you to click on suspicious links.*

    **Non-Spam SMS Examples:**
    - **Example 1:** 
      ```
      Hi John, just a reminder about our meeting tomorrow at 10 AM. See you then!
      ```
      *A straightforward reminder about a scheduled meeting, which is relevant and expected communication.*

    - **Example 2:** 
      ```
      Hi Sarah, can you send me the notes from the meeting last week? Thanks!
      ```
      *A request for information related to a previous meeting, which is a normal, personal communication.*
    """)

    st.subheader('🔍 Classification Report')
    st.write("Here is the classification report showing how the model performs on the real-world examples:")

    # Display classification report heatmap image
    img_path=abs_image_path('real.PNG')
    st.write(f"Image Path:{img_path}")
    img=Image.open(img_path)
    st.image(img, caption='Classification Report Heatmap', use_column_width=True)