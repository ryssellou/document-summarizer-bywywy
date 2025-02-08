import streamlit as st
from transformers import pipeline
import PyPDF2  # For handling PDF files

# Custom CSS for pink theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffcccb;  /* Light pink background */
    }
    h1 {
        color: #e75480;  /* Dark pink for title */
    }
    .stButton button {
        background-color: #e75480;  /* Dark pink for button */
        color: white;
    }
    .stSlider div {
        color: #e75480;  /* Dark pink for slider */
    }
    .stRadio div {
        color: #e75480;  /* Dark pink for radio buttons */
    }
    .stTextArea textarea {
        background-color: #FF69b4;  /* Light pink for text area */
    }
    .stFileUploader div {
        color: #e75480;  /* Dark pink for file uploader */
    }
    .stWarning {
        color: #e75480;  /* Dark pink for warnings */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load summarization model using the new caching function
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
        # Extract text from PDF
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        input_text = ""
        for page in pdf_reader.pages:
            input_text += page.extract_text()
        st.write("PDF text extracted successfully!")

# Summary length slider
summary_length = st.slider("Select summary length (in words):", 50, 200, 100)

# Generate summary
if st.button("Summarize"):
    if input_text:
        with st.spinner("Generating summary..."):
            try:
                # Split text into chunks if it's too long (BART has a token limit)
                max_input_length = 1024  # BART's max token limit
                if len(input_text) > max_input_length:
                    st.warning("The document is too long. Summarizing the first 1024 tokens...")
                    input_text = input_text[:max_input_length]

                # Generate summary
                summary = summarizer(input_text, max_length=summary_length, min_length=int(summary_length / 2), do_sample=False)
                st.subheader("📝 Summary:")
                st.write(summary[0]['summary_text'])
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter or upload some text to summarize.")
