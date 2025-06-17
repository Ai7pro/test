from flask import Flask, request, render_template_string, redirect, url_for, flash
import PyPDF2
import requests
import os
from dotenv import load_dotenv
import re
from langdetect import detect

# Load environment variables
load_dotenv()

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flashing messages

UPLOAD_FORM = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Document Analysis Tool</title>
  <style>
    body {
      background: #181a1b;
      color: #e8e6e3;
      font-family: 'Segoe UI', Arial, sans-serif;
      margin: 0;
      padding: 0;
      min-height: 100vh;
    }
    .container {
      max-width: 700px;
      margin: 40px auto;
      background: #23272b;
      border-radius: 12px;
      box-shadow: 0 4px 24px #000a;
      padding: 32px 32px 24px 32px;
    }
    h2, h3, h4 {
      color: #f3b13c;
      margin-top: 0;
    }
    input[type="file"] {
      background: #23272b;
      color: #e8e6e3;
      border: 1px solid #444;
      border-radius: 6px;
      padding: 8px;
      margin-bottom: 16px;
    }
    input[type="submit"] {
      background: #f3b13c;
      color: #23272b;
      border: none;
      border-radius: 6px;
      padding: 10px 24px;
      font-weight: bold;
      cursor: pointer;
      transition: background 0.2s;
    }
    input[type="submit"]:hover {
      background: #ffcc66;
    }
    ul {
      color: #ff6b6b;
      padding-left: 20px;
    }
    .summary-block {
      background: #181a1b;
      color: #e8e6e3;
      border-radius: 8px;
      padding: 16px;
      font-size: 1.08em;
      margin-bottom: 24px;
      white-space: pre-line;
      box-shadow: 0 2px 8px #0004;
    }
    .points-block {
      background: #23272b;
      color: #e8e6e3;
      border-radius: 8px;
      padding: 16px 24px;
      font-size: 1.05em;
      margin-bottom: 8px;
      white-space: pre-line;
      box-shadow: 0 2px 8px #0004;
    }
    .point-label {
      color: #f3b13c;
      font-weight: bold;
      margin-right: 8px;
      display: inline-block;
      min-width: 220px;
    }
    .spinner {
      display: none;
      margin: 24px auto;
      border: 8px solid #23272b;
      border-top: 8px solid #f3b13c;
      border-radius: 50%;
      width: 60px;
      height: 60px;
      animation: spin 1s linear infinite;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  </style>
  <script>
    function showSpinner() {
      document.getElementById('spinner').style.display = 'block';
    }
    window.onload = function() {
      var form = document.getElementById('uploadForm');
      if (form) {
        form.onsubmit = function() {
          showSpinner();
        };
      }
    };
  </script>
</head>
<body>
  <div class="container">
    <h2>Document Analysis Tool</h2>
    <form id="uploadForm" method=post enctype=multipart/form-data>
      <input type=file name=document accept=".pdf,.txt" required>
      <input type=submit value=Upload>
    </form>
    <div id="spinner" class="spinner"></div>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <ul>
        {% for message in messages %}
          <li>{{ message }}</li>
        {% endfor %}
        </ul>
      {% endif %}
    {% endwith %}
    {% if doc_summary %}
      <h3>Document Summary</h3>
      <div class="summary-block">{{ doc_summary }}</div>
    {% endif %}
    {% if points %}
      <h3>Key Extracted Points</h3>
      <div class="points-block">{{ points|safe }}</div>
      {% if uploaded_filename %}
        <div style="margin-top:16px; color:#f3b13c;">Uploaded file: <b>{{ uploaded_filename }}</b></div>
      {% endif %}
      <script>document.getElementById('spinner').style.display = 'none';</script>
    {% endif %}
  </div>
</body>
</html>
'''

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def clean_text(text):
    """Remove non-English characters and clean the text."""
    # Remove special characters and keep only English text
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', text)
    return cleaned_text

def get_llm_summary(text):
    """Get summary from OpenRouter LLM."""
    prompt = "Summarize this document in a concise paragraph."
    data = {
        "model": "deepseek-ai/deepseek-r1-0528-qwen3-8b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
            {"role": "user", "content": prompt + "\n" + text}
        ]
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error getting LLM response: {str(e)}"

def get_llm_points(text):
    prompt = (
        "Extract the following information from the document. For each, output the field and value in this format:\n"
        "1. Tender Number or Bid number: <value>\n"
        "2. Name of the Work or Searched Result generated in GeMARPTS: <value>\n"
        "3. Department Name: <value>\n"
        "4. ECV (Estimated Contract Value): <value>\n"
        "5. Contract Period / Bid Offer Validity: <value>\n"
        "6. EMD (Earnest Money Deposit): <value>\n"
        "7. EMD Exemption: <value>\n"
        "8. Mode of Payment: <value>\n"
        "9. Eligibility Criteria: <value>\n"
        "If a value is not found, put a dash."
    )
    data = {
        "model": "deepseek-ai/deepseek-r1-0528-qwen3-8b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that extracts information from tender documents."},
            {"role": "user", "content": prompt + "\n" + text}
        ]
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error getting LLM response: {str(e)}"

def parse_points(llm_output):
    points_labels = [
        "1. Tender Number or Bid number",
        "2. Name of the Work or Searched Result generated in GeMARPTS",
        "3. Department Name",
        "4. ECV (Estimated Contract Value)",
        "5. Contract Period / Bid Offer Validity",
        "6. EMD (Earnest Money Deposit)",
        "7. EMD Exemption",
        "8. Mode of Payment",
        "9. Eligibility Criteria"
    ]
    points = {label: '-' for label in points_labels}
    for line in llm_output.splitlines():
        line = line.replace('*', '').strip()
        for label in points_labels:
            if line.lower().startswith(label.lower()):
                value = line[len(label):].strip(' :.-')
                points[label] = value if value else '-'
    return [(label, points[label]) for label in points_labels]

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    points = None
    uploaded_filename = None
    if request.method == 'POST':
        if 'document' not in request.files or request.files['document'].filename == '':
            flash('No file was uploaded.')
        else:
            file = request.files['document']
            uploaded_filename = file.filename
            if file and (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
                try:
                    if file.filename.endswith('.pdf'):
                        text = extract_text_from_pdf(file)
                    else:
                        text = file.read().decode('utf-8')
                    cleaned_text = clean_text(text)
                    points_output = get_llm_points(cleaned_text)
                    points = parse_points(points_output)
                except Exception as e:
                    flash(f'Error processing file: {str(e)}')
            else:
                flash('Unsupported file type. Please upload a PDF or TXT file.')
    # Format points as HTML for display: label in bold, value normal
    points_html = None
    if points:
        points_html = ''.join([
            f'<div style="margin-bottom:10px;"><span class="point-label">{label}:</span> <span style="font-weight:normal;">{value}</span></div>'
            for label, value in points
        ])
    return render_template_string(UPLOAD_FORM, points=points_html, uploaded_filename=uploaded_filename)

if __name__ == '__main__':
    app.run(debug=True) 