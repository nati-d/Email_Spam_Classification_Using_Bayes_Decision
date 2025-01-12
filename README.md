# Spam Message Classifier using Bayes Decision Theory 

## Overview

This project implements a spam message classifier using **Naive Bayes** with **Laplace Smoothing**. The goal is to classify email messages as either **spam** or **non-spam** based on the text content. The Naive Bayes algorithm is a probabilistic classifier that leverages Bayes' Theorem, and it works particularly well for text classification tasks.

In this implementation, we use **Laplace Smoothing** to avoid zero probabilities for unseen words. This ensures that even if a word does not appear in the training data, the model can still make predictions without assigning zero probability to those unseen words.

## Why Naive Bayes?

Naive Bayes is particularly suited for text classification for the following reasons:

1. **Handles Missing Words**: 
   - In real-world applications, the training data may not contain all possible words. Naive Bayes assigns a small non-zero probability to unseen words using Laplace Smoothing, preventing the classifier from failing when encountering unknown terms.

2. **Zero Probability Prevention**: 
   - Traditional **Maximum Likelihood Estimation (MLE)** may assign zero probability to unseen words, which can severely affect the model's performance. Naive Bayes with Laplace Smoothing ensures that no word has a probability of zero, even if it hasn't appeared in the training data.

3. **Scalability**: 
   - Naive Bayes is computationally efficient and works well with large datasets. It's particularly effective when the dataset is sparse, which is often the case with text data, where many words may not appear in every message.

4. **Simplified Assumption**: 
   - The Naive Bayes model assumes that the presence of a word in a message is independent of the presence of any other words. This assumption simplifies the computation of probabilities, and although it may not always be accurate, it has proven to work well in practice.

## How It Works

The classifier works in the following steps:

1. **Data Cleaning**:
   - The raw email messages are first preprocessed by converting all text to lowercase, removing punctuation, and splitting the text into individual words (tokens). This helps standardize the text and make it easier to analyze.

2. **Word Frequency Calculation**:
   - The frequency of each word in the **spam** and **non-spam** messages is calculated. This step helps determine how often each word appears in each class of messages.

3. **Laplace Smoothing**:
   - To avoid zero probabilities, **Laplace Smoothing** is applied. This involves adding a small constant (usually 1) to the word counts and adjusting the total word counts in both classes. This ensures that words not seen in the training data still have a small non-zero probability.

4. **Calculating Likelihoods**:
   - For each word, the likelihood of it being in a spam or non-spam message is calculated using the smoothed word frequencies. These likelihoods are used to make predictions.

5. **Prior Probability**:
   - The prior probability of a message being spam or non-spam is computed based on the proportions of each class in the dataset.

6. **Prediction**:
   - For each new message, the classifier calculates the probability of the message being spam or non-spam by combining the prior probabilities with the likelihoods of the words in the message. The class with the higher probability is selected as the predicted label.

## Data Preprocessing

- **Cleaning**: The text data is cleaned to remove noise such as punctuation, case sensitivity, and unnecessary characters.
- **Tokenization**: The cleaned text is split into individual words (tokens).
- **Labeling**: Each email message is labeled as either **spam** or **non-spam** (often called **ham**).

## Model Implementation

1. **Data Loading**:
   - The dataset is loaded from a text file, where each message is separated by a tab (`\t`) and labeled with either "spam" or "ham" (non-spam).
   
2. **Word Frequency Calculation**:
   - Word frequencies are calculated for both spam and non-spam messages.
   
3. **Likelihood Calculation**:
   - Using Laplace smoothing, the likelihoods for each word being spam or non-spam are computed.

4. **Prior Probability**:
   - The prior probability of a message being spam or non-spam is computed based on the proportions of each class in the dataset.

5. **Prediction**:
   - Given a set of cleaned email messages, the model will classify each message as spam or non-spam by calculating the likelihoods and comparing them.

## Key Concepts in the Model

- **Laplace Smoothing**: 
   - Used to handle the problem of zero probabilities. It ensures that no word has a zero probability, even if it is not present in the training data.
   
- **Prior Probabilities**: 
   - The likelihood of a message being spam or non-spam before considering the words in the message.

- **Likelihoods**: 
   - The probability of a word occurring in a message of a given class (spam or non-spam).

## Usage

1. **Data Input**:
   - The dataset must be in a format where each line contains a label (spam/ham) and the corresponding email message. The messages should be separated by a tab (`\t`), and the first line should be ignored as it contains no headers.

2. **Prediction**:
   - Given a set of cleaned email messages, the model will classify each message as spam or non-spam by calculating the likelihoods and comparing them.


## Results and Evaluation

- **Spam Accuracy**: 
   - The model accurately classifies spam messages based on the likelihoods of individual words.
   
- **Performance Metrics**: 
   - The model can be evaluated using common classification metrics such as accuracy, precision, recall, and F1-score.

## Conclusion

This project demonstrates the power of **Bayes Decision Theory Using Naive Bayes Algorithm** in text classification tasks, particularly spam detection. By utilizing smoothing, the model is robust to unseen words, making it practical for real-world applications where new or unseen words often appear.
