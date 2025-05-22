from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from datasets import load_dataset

# Load the SpamAssassin dataset from HuggingFace
print("Loading SpamAssassin dataset...")
ds = load_dataset("talby/spamassassin", "text")

# Convert dataset to pandas DataFrame
df = pd.DataFrame({
    'text': ds['train']['text'],
    'label': ds['train']['label'],
    'group': ds['train']['group']
})

# Check the label-class mapping
print("Label distribution by group:")
print(df.groupby('label')['group'].value_counts())

# Sanity check for class presence
assert set(df['label'].unique()) == {0, 1}, "Labels must be 0 (ham) or 1 (spam)"

# Split the data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    test_texts, test_labels, test_size=0.5, stratify=test_labels, random_state=42)

# Verify class distribution
print(f"Training set class distribution: {pd.Series(train_labels).value_counts().to_dict()}")
print(f"Validation set class distribution: {pd.Series(val_labels).value_counts().to_dict()}")
print(f"Test set class distribution: {pd.Series(test_labels).value_counts().to_dict()}")

# Build the pipeline (toggle TF-IDF on/off as needed)
pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words='english')),
    # ('tfidf', TfidfTransformer()),  # Uncomment to use TF-IDF
    ('clf', MultinomialNB(alpha=1.0))
])

# Train the model
pipeline.fit(train_texts, train_labels)

# Predict on validation set
prob_preds = pipeline.predict_proba(val_texts)

# Safe entropy computation per row
entropy_scores = np.apply_along_axis(lambda p: entropy(p, base=2), axis=1, arr=prob_preds)

# Predictions
val_preds = pipeline.predict(val_texts)

# Create results DataFrame
val_df = pd.DataFrame({
    'text': val_texts,
    'true_label': val_labels,
    'predicted': val_preds,
    'prob_spam': prob_preds[:, 1],
    'entropy': entropy_scores
})

# Top uncertain predictions
high_uncertainty = val_df.sort_values('entropy', ascending=False).head(5)
print("\nHighest uncertainty predictions:")
for i, row in high_uncertainty.iterrows():
    print(f"Text: {row['text'][:100].strip()}...")
    print(f"True: {'spam' if row['true_label'] == 1 else 'ham'}, " +
          f"Predicted: {'spam' if row['predicted'] == 1 else 'ham'}, " +
          f"P(spam): {row['prob_spam']:.4f}, Entropy: {row['entropy']:.4f}")
    print("-" * 80)

# Misclassified examples
misclassified = val_df[val_df['true_label'] != val_df['predicted']]
if len(misclassified) > 0:
    print("\nMisclassified examples:")
    for i, row in misclassified.head(5).iterrows():
        print(f"Text: {row['text'][:100].strip()}...")
        print(f"True: {'spam' if row['true_label'] == 1 else 'ham'}, " +
              f"Predicted: {'spam' if row['predicted'] == 1 else 'ham'}, " +
              f"P(spam): {row['prob_spam']:.4f}")
        print("-" * 80)
else:
    print("\nNo misclassified examples in validation set.")

# Classification report
target_names = ["ham", "spam"]
print("\nClassification Report:")
print(classification_report(val_labels, val_preds, target_names=target_names))

# Entropy distribution plot
plt.figure(figsize=(10, 6))
plt.hist(entropy_scores, bins=30)
plt.title("Predictive Entropy Distribution")
plt.xlabel("Entropy")
plt.ylabel("Number of Email Messages")
plt.grid(True, alpha=0.3)
plt.savefig('entropy_distribution.png')
plt.show()

# Confusion matrix
cm = confusion_matrix(val_labels, val_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, 
            yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Feature importance
def get_top_features(pipeline, class_label, n=15):
    feature_names = pipeline.named_steps['vect'].get_feature_names_out()
    coefs = pipeline.named_steps['clf'].feature_log_prob_[class_label]
    top_indices = np.argsort(coefs)[-n:]
    return [(feature_names[i], coefs[i]) for i in top_indices]

print("\nTop features for spam detection:")
for feature, score in reversed(get_top_features(pipeline, 1)):
    print(f"{feature}: {score:.4f}")

print("\nTop features for ham detection:")
for feature, score in reversed(get_top_features(pipeline, 0)):
    print(f"{feature}: {score:.4f}")
