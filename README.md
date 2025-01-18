# MentalHealthDisorderClassification
## Project Overview

Mental health disorders affect millions globally, often going undiagnosed or misdiagnosed due to stigma, lack of resources, and challenges in detection. This project aims to address these issues by leveraging advanced Natural Language Processing (NLP) and machine learning techniques to develop a robust system that classifies textual data into seven predefined mental health categories: Normal, Depression, Suicidal, Anxiety, Stress, Bipolar, and Personality Disorders.

By analyzing linguistic patterns and emotional cues present in text data, the system seeks to aid early detection, enabling timely interventions that can improve outcomes for individuals at risk. Potential real-world applications include clinical decision support systems, social media monitoring for crisis intervention, and intelligent therapeutic chatbots.

## Objectives

- Accurate Classification: Differentiate between normal emotional expressions and indicators of mental health conditions.
- Context Sensitivity: Capture linguistic nuances for precise classification.
- Timely Intervention: Facilitate early detection to improve outcomes.


## Dataset
* Sources: Social media platforms like Reddit and X (formerly Twitter) and other textual data repositories.
* Categories: The dataset includes labeled data for seven mental health statuses.
* Preparation:
    * Text tokenization using NLTK.
    * Pre-trained GloVe embeddings to represent linguistic patterns in a high-dimensional space.
    * Splitting into training and test sets (80:20 ratio) for evaluation.


## Approaches

1. Bayesian Neural Networks (BNNs):

    * Incorporates uncertainty estimation, enhancing robustness in handling ambiguous or noisy text data.
    * Architecture:
        * Three Bayesian linear layers with prior distributions.
        * Two ReLU-activated hidden layers.
        * An output layer for classification.
    * Performance: Achieved a test accuracy of 76.42% after 50 epochs with progressive loss reduction.


2. Long Short-Term Memory (LSTM) Networks:
    * Captures sequential and contextual language patterns.
    * Bidirectional LSTM layers enable understanding of context from both forward and backward directions.
    * Dropout regularization was applied to prevent overfitting.
    * Performance: Achieved a peak test accuracy of 77.14% after 10 epochs.

## Training and Evaluation

    1 Embedding Techniques: Used pre-trained GloVe embeddings to represent text.
    2 Optimizer: Adam optimizer with a learning rate of 0.001.
    3 Loss Function: Cross-Entropy Loss.
    4 Metrics: Accuracy and loss reduction over epochs.
    5 Tools: Implemented using PyTorch, with custom Dataset and DataLoader classes for efficient handling of text sequences.
.

## Results
* The BNN model demonstrated strong performance, with uncertainty estimation providing added interpretability and reliability, particularly in handling noisy data.

* The LSTM model excelled in capturing sequential and contextual relationships, slightly outperforming BNN in test accuracy.

* Both models exhibited consistent accuracy across different epochs and showed potential for real-world applications.

