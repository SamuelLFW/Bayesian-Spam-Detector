# Bayesian-Spam-Detector - A Spam Email Classifier using Naive Bayes

This project builds a spam email classification pipeline using the **SpamAssassin** dataset and a **Multinomial Naive Bayes** classifier. It demonstrates text preprocessing, feature extraction via Bag-of-Words and TF-IDF, and classification evaluation with scikit-learn.

## 🚀 Features

- Load and explore the [SpamAssassin dataset](https://huggingface.co/datasets/talby/spamassassin) via HuggingFace Datasets
- Email text preprocessing (HTML tag removal, email header stripping, etc.)
- Vectorization with `CountVectorizer` and `TfidfVectorizer`
- Training a Multinomial Naive Bayes classifier
- Evaluation with:
  - Accuracy
  - Confusion matrix
  - Classification report

## 🧰 Requirements

Make sure you have the following Python packages installed:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn datasets huggingface_hub
```

## 📁 Project Structure

```
spam_classifier/
├── spam_classifier.ipynb     # Main notebook for preprocessing, training, and evaluation
├── README.md                 # Project description and instructions
```

## 📝 Dataset

- **Source**: [SpamAssassin Dataset on HuggingFace](https://huggingface.co/datasets/talby/spamassassin)
- **Labels**: `1` for spam, `0` for ham

## 🧪 Example Usage

The notebook walks through the following:

1. Load and inspect the dataset
2. Clean and preprocess the email text
3. Split data into training and test sets
4. Extract features using `CountVectorizer` and `TfidfVectorizer`
5. Train Naive Bayes classifier
6. Evaluate and visualize results

## 📊 Output Preview

- Model accuracy and precision-recall report
- Confusion matrix heatmap via seaborn
- Distribution of label classes
