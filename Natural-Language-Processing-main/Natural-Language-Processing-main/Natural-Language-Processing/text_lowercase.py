# Step 1: Import the necessary library
import pandas as pd

# Step 2: Create or load your dataset
# Here, we are creating a sample dataset for demonstration
data = {
    'Text': [
        "This is a Sample TEXT to be Converted.",
        "Natural Language Processing is AWESOME!",
        "LOWER CASE everything properly.",
        "Machine learning models can be complex.",
        "Deep learning requires large datasets.",
        "Data preprocessing is an important step.",
        "Text data often needs cleaning.",
        "Tokenization splits text into words or sentences.",
        "Lemmatization reduces words to their base forms.",
        "Punctuation and special characters are often removed.",
        "Stop words like 'the' or 'is' are usually ignored.",
        "Stemming cuts words down to their root form.",
        "Feature extraction transforms text into useful inputs.",
        "The bag-of-words model is a simple text representation.",
        "TF-IDF helps weigh important words in documents.",
        "Word embeddings capture word meanings in context.",
        "Neural networks can be used for text classification.",
        "Recurrent neural networks handle sequential data.",
        "Attention mechanisms help focus on important parts of the text.",
        "Transformers have revolutionized natural language processing.",
        "BERT is a powerful model for understanding context.",
        "GPT models generate human-like text.",
        "Language models are pre-trained on vast amounts of data.",
        "Transfer learning applies knowledge from one task to another.",
        "Supervised learning requires labelled data for training.",
        "Unsupervised learning finds hidden patterns in data."
    ]
}

# Step 3: Create a pandas DataFrame
df = pd.DataFrame(data)

# Step 4: Apply lowercasing to the text column
df['Lowercased_Text'] = df['Text'].str.lower()

# Step 5: View the original and lowercased text side by side
print(df[['Text', 'Lowercased_Text']])

# Step 6: (Optional) Save the cleaned dataset to a CSV file
df.to_csv('cleaned_data.csv', index=False)