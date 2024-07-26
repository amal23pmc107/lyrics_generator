import pandas as pd
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['lyrics'] = df['lyrics'].str.replace('\n', ' ', regex=True)  # Remove newline characters
    return df

# Generate lyrics using the pre-trained model
def generate_lyrics(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs, 
        max_length=max_length, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2
    )
    lyrics = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return lyrics

# Streamlit app
def main():
    st.title("Lyrics Generator")

    # Load the pre-trained model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    st.header("Enter Keywords to Generate Lyrics")
    keywords = st.text_input("Keywords (comma-separated)", "")

    # Load and preprocess the dataset
    dataset_file_path = 'lyrics_dataset.csv'
    df = load_and_preprocess_data(dataset_file_path)
    st.write("Dataset preview:")
    st.write(df.head())

    if st.button("Generate Lyrics"):
        if keywords:
            prompt = " ".join([kw.strip() for kw in keywords.split(",")])
            lyrics = generate_lyrics(model, tokenizer, prompt)
            st.subheader("Generated Lyrics")
            st.write(lyrics)
        else:
            st.write("Please enter some keywords.")

if __name__ == "__main__":
    main()
