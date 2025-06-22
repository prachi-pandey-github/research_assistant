📄 Document-Aware AI Assistant with Gemini
This Streamlit app enables users to upload PDF or TXT documents, receive concise AI-generated summaries, ask questions with contextual answers and justifications, and test their comprehension through intelligent challenge questions.

🚀 Setup Instructions
1. Clone the repository
git clone https://github.com/your-username/document-ai-assistant.git
cd document-ai-assistant

2. Create a virtual environment
   
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Run the app
streamlit run app.py

🧠 Architecture & Reasoning Flow
🔹 1. Document Upload & Preprocessing
Users upload a .pdf or .txt document.

Text is extracted using PyPDF2 (PDF) or standard UTF-8 decoding (TXT).

Text is cleaned, split into paragraphs and sentences, and analyzed for word/character count.

🔹 2. Summary Generation
The full document text is passed to Gemini.

Gemini generates a concise summary (≤150 words).

Summary is displayed alongside key metrics (word count, character count, etc.).

🔹 3. Ask Anything Mode
Users enter free-form questions about the uploaded document.

Gemini provides:

✅ Direct Answer

📌 Justification (how it was derived)

📖 Source Snippet (quote from document)

🔹 5. Challenge Me Mode
Gemini creates 3 logic-driven questions:

One each for comprehension, analysis, and inference.

Users answer and receive:

📊 AI-evaluated score (0–100)

💬 Constructive feedback

🔍 Justification and reference from the document

🔹 6. Session Management
The app maintains state for:

Uploaded document

Assistant instance

Interaction history

Challenge questions

📁 File Structure
bash
Copy
Edit
├── app.py                  # Main Streamlit app      
├── requirements.txt        # Python dependencies
└── README.md
📦 Features
Upload PDF or TXT files

Summarize documents in plain English

Ask intelligent questions with answer justifications

Evaluate your understanding via AI-generated challenges

Clean and interactive Streamlit interface
