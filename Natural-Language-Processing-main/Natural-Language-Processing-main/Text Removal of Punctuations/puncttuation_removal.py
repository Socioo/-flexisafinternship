# Step 1: Import the necessary libraries
import string
import pandas as pd

# Step 2: Create or load your dataset
# We will use a sample dataset with sentences containing various punctuations
data = {
    'Text': [
        "Hello, World! This is an example sentence.",
        "Natural Language Processing is interesting!!!",
        "What's your plan for today? #AI #ML",
        "Python is great; however, it requires practice.",
        "Are you ready to remove punctuations? Let's do it!",
        "Data science: It's fun, but also challenging.",
        "When life gives you lemons, make lemonade!",
        "Programming in Python? Sure! Why not.",
        "Punctuations, like commas, periods, and exclamation marks, can be removed.",
        "Do you love NLP? If so, you'll love this project."
    ]
}

# Step 3: Create a pandas DataFrame
df = pd.DataFrame(data)

# Step 4: Define the function to remove punctuations
df['No_Punctuations'] = df['Text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# Step 5: View the original and punctuation-removed text side by side
print(df[['Text', 'No_Punctuations']])

# Step 6: (Optional) Save the cleaned dataset to a CSV file
df.to_csv('cleaned_text.csv', index=False)