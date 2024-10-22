import pandas as pd
import emoji

# Larger dataset with various emojis
data = {
    'Text': [
        "I am so happy today 😊",
        "She looks sad 😢",
        "That was a funny joke 😂",
        "I am feeling confused 😕",
        "It's a big surprise 😲",
        "Oh no! I'm shocked 😱",
        "This is amazing ❤️",
        "He is angry 😡",
        "That made me laugh 🤣",
        "I don't know what to say 😐",
        "She just gave me a gift 🎁",
        "They are celebrating 🎉",
        "It's raining heavily 🌧️",
        "Good morning! ☀️",
        "I love ice cream 🍦",
        "Let's go for a ride 🚗",
        "He is studying 📚",
        "They are travelling by plane ✈️",
        "Cooking dinner 🍲",
        "Time to sleep 😴"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Function to convert emojis to words using emoji library
def convert_emoji_to_words(text):
    return emoji.demojize(text, delimiters=(" ", " "))

# Apply the function to the dataset
df['Converted_Text'] = df['Text'].apply(convert_emoji_to_words)

# Display the original and converted text
print(df[['Text', 'Converted_Text']])

# (Optional) Save the DataFrame to a CSV file
df.to_csv('emoji_to_words.csv', index=False)