import streamlit as st
import nltk
from PyPDF2 import PdfReader

# Step 3: Ensure NLTK sentence tokenizer is available
nltk.download("punkt", quiet=True)

st.set_page_config(page_title="Text Chunking using NLTK", layout="wide")

st.title("PDF Sentence Chunker Demo (NLTK Sentence Tokenizer)")
st.write(
    "Upload a PDF file, extract text, and split it into sentences using "
    "NLTK's `sent_tokenize`"
)

# Step 1: Upload and read PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    try:
        reader = PdfReader(uploaded_file)

        # Step 2: Extract text from PDF
        extracted_text = []
        for page in reader.pages:
            text = page.extract_text() or ""
            extracted_text.append(text)

        full_text = " ".join(extracted_text).strip()

        st.subheader("PDF Information")
        st.write(f"Number of pages: **{len(reader.pages)}**")
        st.write(f"Total characters extracted: **{len(full_text)}**")

        if not full_text:
            st.warning("No text could be extracted from the PDF.")
        else:
            # Step 3: Sentence tokenization (preprocessing)
            sentences = nltk.sent_tokenize(full_text)
            st.success(f"Total sentences detected: {len(sentences)}")

            # Step 3: Display sample sentences (index 58–68)
            st.subheader("Sample Extracted Sentences (Index 58 to 68)")
            start_index = 58
            end_index = 68

            if len(sentences) >= end_index:
                for i in range(start_index, end_index):
                    st.markdown(f"**{i}.** {sentences[i]}")
            else:
                st.warning(
                    "The document does not contain enough sentences to display indices 58–68."
                )

            # Step 4: Semantic sentence chunking result
            st.subheader("Semantic Sentence Chunking Output")
            st.write(
                "Each sentence represents a semantic chunk, allowing meaningful "
                "information to be processed independently."
            )

            for i, sentence in enumerate(sentences[start_index:end_index], start=start_index):
                st.markdown(f"🔹 **Chunk {i}**: {sentence}")

            with st.expander("Show raw extracted text (first 2000 characters)"):
                st.text(full_text[:2000])

    except Exception as e:
        st.error(f"Error processing PDF file: {e}")
else:
    st.info("Please upload a PDF to begin.")
