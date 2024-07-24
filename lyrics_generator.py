import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load the pre-trained model and tokenizer
model_name = "ECE1786-AG/lyrics-generator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set up the text generation pipeline
lyrics_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_lyrics(keywords, max_length=100, num_return_sequences=1):
    # Format the input keywords into a prompt
    prompt = " ".join(keywords)
    
    # Generate lyrics based on the prompt
    generated_lyrics = lyrics_generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
    
    # Extract and return the generated text
    return [lyrics["generated_text"] for lyrics in generated_lyrics]

# Streamlit app
st.title("Lyrics Generator")

# Input text
keywords_input = st.text_input("Enter keywords for generating lyrics (separated by spaces):")

# Parameters for text generation
max_length = st.slider("Max length of lyrics", min_value=10, max_value=500, value=100)
num_return_sequences = st.slider("Number of generated sequences", min_value=1, max_value=5, value=1)

# Generate lyrics when button is clicked
if st.button("Generate Lyrics"):
    if keywords_input:
        keywords = keywords_input.split()
        lyrics = generate_lyrics(keywords, max_length=max_length, num_return_sequences=num_return_sequences)
        st.subheader("Generated Lyrics")
        for idx, lyric in enumerate(lyrics):
            st.text_area(f"Lyrics {idx+1}", value=lyric, height=300)
    else:
        st.warning("Please enter some keywords to generate lyrics.")
