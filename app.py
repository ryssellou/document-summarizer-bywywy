import streamlit as st
from transformers import pipeline
import PyPDF2  # For handling PDF files
import gc  # Garbage Collector

# Custom CSS for professional sleek theme with pink and black palette
st.markdown(
    """
    <style>
    .stApp { background-color: #1e1e1e; color: #ffffff; }
    h1 { color: #ff4081; }
    .stButton button { background-color: #ff4081; color: white; border-radius: 8px; }
    .stSlider div { color: #ff4081; }
    .stRadio div { color: #ff4081; }
    .stTextArea textarea { background-color: #2c2c2c; color: white; border-radius: 5px; }
    .stFileUploader div { color: #ff4081; }
    .stWarning { color: #ff4081; }
    .stSpinner { color: #ff4081; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load summarization model efficiently
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Streamlit app
st.title("📄 PDF Summarizer by Ryry")
st.write("Upload a PDF or paste text, and get a concise summary!")

# Input options
input_option = st.radio("Choose input type:", ("Paste Text", "Upload PDF"))
input_text = ""

if input_option == "Paste Text":
    input_text = st.text_area("Paste your text here:", height=200)
elif input_option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])
    if uploaded_file is not None:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        input_text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        st.write("PDF text extracted successfully!")

# Limit input size
max_input_length = 1024  # BART's max token limit
if len(input_text) > max_input_length:
    st.warning("The input text is too long. Trimming to 1024 characters...")
    input_text = input_text[:max_input_length]

# Summary length slider
summary_length = st.slider("Select summary length (in words):", 50, 150, 100)

# Generate summary
if st.button("Summarize"):
    if input_text:
        with st.spinner("Generating summary..."):
            try:
                summary = summarizer(input_text, max_length=summary_length, min_length=int(summary_length / 2), do_sample=False)
                st.subheader("📝 Summary:")
                st.write(summary[0]['summary_text'])
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter or upload some text to summarize.")

# Free up unused memory
gc.collect()
