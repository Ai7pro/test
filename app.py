import streamlit as st
import os
import re
import requests
import tempfile
import nltk

# Download required NLTK resources at runtime
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from PyPDF2 import PdfReader
import docx2txt

def process_file(uploaded_file):
    """Process different file formats and extract text"""
    text = ""
    extension = uploaded_file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name
    try:
        if extension == 'pdf':
            reader = PdfReader(file_path)
            # Only extract text from pages where text is found
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif extension in ['docx', 'doc']:
            text = docx2txt.process(file_path)
        elif extension == 'txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
    finally:
        os.unlink(file_path)
    return text

def extract_links(text):
    """Extract hyperlinks from text"""
    url_pattern = r'(https?://\S+)'
    return re.findall(url_pattern, text)

def fetch_link_content(url):
    """Fetch content from a URL"""
    try:
        response = requests.get(url, timeout=10)
        return response.text[:5000]  # Return first 5000 characters
    except:
        return None

def sentence_similarity(sent1, sent2, stopwords=None):
    """Helper function for similarity calculation"""
    if stopwords is None:
        stopwords = []
    words1 = [w.lower() for w in nltk.word_tokenize(sent1)]
    words2 = [w.lower() for w in nltk.word_tokenize(sent2)]
    all_words = list(set(words1 + words2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for w in words1:
        if w not in stopwords:
            vector1[all_words.index(w)] += 1
    for w in words2:
        if w not in stopwords:
            vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)

def generate_summary(text, top_n=5):
    """Generate summary using TextRank algorithm"""
    stop_words = stopwords.words('english')
    sentences = nltk.sent_tokenize(text)
    if len(sentences) == 0:
        return "No valid sentences found for summarization."
    if len(sentences) < top_n:
        top_n = len(sentences)
    # Similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(
                    sentences[i], sentences[j], stop_words)
    # PageRank algorithm
    nx_graph = nx.from_numpy_array(similarity_matrix)
    try:
        scores = nx.pagerank(nx_graph)
    except Exception as e:
        return f"Summary generation failed: {e}"
    ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    return " ".join([ranked[i][1] for i in range(top_n)])

def main():
    st.title("AI Document Scrutiny Tool")
    uploaded_file = st.file_uploader("Upload Document", type=['pdf', 'docx', 'doc', 'txt'])

    if uploaded_file:
        text = process_file(uploaded_file)
        st.write(f"**Extracted text length:** {len(text)}")
        st.write("**Preview of extracted text:**")
        st.write(text[:500])

        if not text or len(text.strip()) == 0:
            st.error("No text could be extracted from the document. Please upload a text-based file (not scanned images or unsupported formats).")
            return

        links = extract_links(text)
        st.subheader("Document Summary")
        summary = generate_summary(text)
        st.write(summary)

        if links:
            st.subheader("Extracted Links")
            for link in links:
                with st.expander(link):
                    content = fetch_link_content(link)
                    if content:
                        st.write(content[:500])  # Show first 500 characters
                    else:
                        st.error("Could not fetch link content")

        st.subheader("Document Q&A")
        query = st.text_input("Ask a question about the document:")
        if query:
            sentences = nltk.sent_tokenize(text)
            relevant_text = [s for s in sentences if query.lower() in s.lower()]
            if relevant_text:
                st.write("**Relevant Information:**")
                st.write("\n".join(relevant_text[:3]))  # Show top 3 relevant sentences
            else:
                st.write("No relevant information found in document.")

if __name__ == "__main__":
    main()
