import streamlit as st
import PyPDF2
import io
import os
import json
import re
from typing import List, Dict, Tuple
import google.generativeai as genai


class DocumentProcessor:
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text content from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def extract_text_from_txt(self, txt_file) -> str:
        """Extract text content from TXT file"""
        try:
            text = txt_file.read().decode('utf-8')
            return text
        except Exception as e:
            st.error(f"Error reading TXT file: {str(e)}")
            return ""
    
    def preprocess_text(self, text: str) -> Dict:
        """Preprocess text and extract basic information"""
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Basic splitting
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        return {
            'full_text': text,
            'sentences': sentences,
            'paragraphs': paragraphs,
            'word_count': len(text.split()),
            'char_count': len(text)
        }

class GeminiDocumentAssistant:
    def __init__(self, processed_data: Dict, api_key: str):
        self.processed_data = processed_data
        self.conversation_history = []
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create a comprehensive document context
        self.document_context = self.processed_data['full_text']
    
    def generate_summary(self, max_words: int = 150) -> str:
        """Generate a summary using Gemini"""
        try:
            prompt = f"""
            Please provide a concise summary of the following document in no more than {max_words} words. 
            Focus on the main topics, key points, and overall purpose of the document.
            
            Document:
            {self.document_context[:4000]}  # Limit context to avoid token limits
            
            Summary (max {max_words} words):
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")
            return "Unable to generate summary. Please check your API key and try again."
    
    def answer_question(self, question: str) -> Dict:
        """Answer a question using Gemini with document context"""
        try:
            prompt = f"""
            You are a GenAI assistant that analyzes user-uploaded documents. Your tasks are:

Answer Questions: Respond to user queries with accurate, concise answers that may require inference.

Ask Logic-Based Questions: Pose reasoning questions about the document and assess user responses.

Justify Every Answer: Support all answers and feedback with direct references from the document (quotes, sections, or page numbers).

Be clear, professional, and avoid assumptions not grounded in the text.
            
            DOCUMENT CONTENT:
            {self.document_context}
            
            QUESTION: {question}
            
            Please provide your response in the following format:
            ANSWER: [Your answer based on the document]
            JUSTIFICATION: [Explain where in the document you found this information]
            SOURCE_SNIPPET: [Quote the relevant portion from the document]
            """
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse the response
            answer = ""
            justification = ""
            source_snippet = ""
            
            lines = response_text.split('\n')
            current_section = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith('ANSWER:'):
                    current_section = "answer"
                    answer = line.replace('ANSWER:', '').strip()
                elif line.startswith('JUSTIFICATION:'):
                    current_section = "justification"
                    justification = line.replace('JUSTIFICATION:', '').strip()
                elif line.startswith('SOURCE_SNIPPET:'):
                    current_section = "source"
                    source_snippet = line.replace('SOURCE_SNIPPET:', '').strip()
                elif line and current_section:
                    if current_section == "answer":
                        answer += " " + line
                    elif current_section == "justification":
                        justification += " " + line
                    elif current_section == "source":
                        source_snippet += " " + line
            
            # If parsing failed, use the full response as answer
            if not answer:
                answer = response_text
                justification = "Generated by AI based on document content"
                source_snippet = "Full document context considered"
            
            # Store in conversation history
            self.conversation_history.append({
                'question': question,
                'answer': answer,
                'justification': justification,
                'source_snippet': source_snippet,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            return {
                'answer': answer,
                'justification': justification,
                'source_snippet': source_snippet
            }
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            return {
                'answer': error_msg,
                'justification': "Error occurred during processing",
                'source_snippet': None
            }
    
    def generate_challenge_questions(self) -> List[Dict]:
        """Generate challenge questions using Gemini"""
        try:
            prompt = f"""
            Based on the document provided below, generate exactly 3 challenging questions that test deep comprehension and logical reasoning. 
            
            Requirements for questions:
            1. Each question should require understanding of the document content
            2. Questions should test different aspects: comprehension, analysis, inference
            3. Avoid simple factual recall - focus on understanding and reasoning
            4. Each question should have a clear connection to specific parts of the document
            
            DOCUMENT CONTENT:
            {self.document_context}
            
            Please provide your response in the following format for each question:
            
            QUESTION_1: [First challenging question]
            TYPE_1: [comprehension/analysis/inference]
            GUIDANCE_1: [Hint for answering this question]
            
            QUESTION_2: [Second challenging question]
            TYPE_2: [comprehension/analysis/inference]
            GUIDANCE_2: [Hint for answering this question]
            
            QUESTION_3: [Third challenging question]
            TYPE_3: [comprehension/analysis/inference]
            GUIDANCE_3: [Hint for answering this question]
            """
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse the questions
            questions = []
            lines = response_text.split('\n')
            
            current_question = {}
            for line in lines:
                line = line.strip()
                if line.startswith('QUESTION_'):
                    if current_question:
                        questions.append(current_question)
                    current_question = {'question': line.split(':', 1)[1].strip() if ':' in line else line}
                elif line.startswith('TYPE_'):
                    current_question['type'] = line.split(':', 1)[1].strip() if ':' in line else 'comprehension'
                elif line.startswith('GUIDANCE_'):
                    current_question['answer_guidance'] = line.split(':', 1)[1].strip() if ':' in line else 'Think carefully about the document content'
            
            if current_question:
                questions.append(current_question)
            
            # Ensure we have exactly 3 questions, create defaults if needed
            while len(questions) < 3:
                questions.append({
                    'question': f"What are the main points discussed in this document? (Question {len(questions) + 1})",
                    'type': 'comprehension',
                    'answer_guidance': 'Consider the overall structure and key themes'
                })
            
            return questions[:3]  # Return exactly 3 questions
            
        except Exception as e:
            st.error(f"Error generating questions: {str(e)}")
            return [
                {
                    'question': "What is the main topic or purpose of this document?",
                    'type': 'comprehension',
                    'answer_guidance': 'Look for the central theme and objective'
                },
                {
                    'question': "What evidence or examples support the main arguments in the document?",
                    'type': 'analysis',
                    'answer_guidance': 'Identify specific supporting details and examples'
                },
                {
                    'question': "What conclusions or implications can be drawn from the information presented?",
                    'type': 'inference',
                    'answer_guidance': 'Think about what the information suggests beyond what is explicitly stated'
                }
            ]
    
    def evaluate_answer(self, question: str, user_answer: str, question_type: str) -> Dict:
        """Evaluate user's answer using Gemini"""
        try:
            prompt = f"""
            You are evaluating a student's answer to a comprehension question based on a document.
            
            DOCUMENT CONTENT:
            {self.document_context}
            
            QUESTION: {question}
            QUESTION TYPE: {question_type}
            
            STUDENT'S ANSWER: {user_answer}
            
            Please evaluate the student's answer and provide:
            1. A score from 0-100 based on accuracy and completeness
            2. Constructive feedback explaining the score
            3. Reference to specific parts of the document that support or contradict the answer
            4. Suggestions for improvement if needed
            
            Provide your response in this format:
            SCORE: [0-100]
            FEEDBACK: [Detailed feedback on the answer quality]
            JUSTIFICATION: [How the answer relates to the document content]
            REFERENCE: [Specific part of document that's relevant]
            """
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse the evaluation
            score = "N/A"
            feedback = ""
            justification = ""
            reference_content = ""
            
            lines = response_text.split('\n')
            current_section = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith('SCORE:'):
                    current_section = "score"
                    score_text = line.replace('SCORE:', '').strip()
                    # Extract numeric score
                    score_match = re.search(r'\b(\d+)\b', score_text)
                    if score_match:
                        score = f"{score_match.group(1)}%"
                    else:
                        score = score_text
                elif line.startswith('FEEDBACK:'):
                    current_section = "feedback"
                    feedback = line.replace('FEEDBACK:', '').strip()
                elif line.startswith('JUSTIFICATION:'):
                    current_section = "justification"
                    justification = line.replace('JUSTIFICATION:', '').strip()
                elif line.startswith('REFERENCE:'):
                    current_section = "reference"
                    reference_content = line.replace('REFERENCE:', '').strip()
                elif line and current_section:
                    if current_section == "feedback":
                        feedback += " " + line
                    elif current_section == "justification":
                        justification += " " + line
                    elif current_section == "reference":
                        reference_content += " " + line
            
            # Provide defaults if parsing failed
            if not feedback:
                feedback = response_text
            if not justification:
                justification = "Evaluated based on document content alignment"
            
            return {
                'feedback': feedback,
                'score': score,
                'justification': justification,
                'reference_content': reference_content
            }
            
        except Exception as e:
            return {
                'feedback': f"Error evaluating answer: {str(e)}",
                'score': "N/A",
                'justification': "Evaluation failed due to technical error",
                'reference_content': "Unable to provide reference"
            }

def get_api_key():
    """Get Gemini API key from Streamlit secrets"""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("🔑 **API Key Not Found**")
        st.error("Please add your Gemini API key to Streamlit secrets.")
        st.info("""
        **To add secrets:**
        
        **For Streamlit Cloud:**
        1. Go to your app settings
        2. Click on "Secrets"
        3. Add: `GEMINI_API_KEY = "your_api_key_here"`
        
        **For Local Development:**
        1. Create `.streamlit/secrets.toml` in your project root
        2. Add: `GEMINI_API_KEY = "your_api_key_here"`
        
        **Get your API key from:** https://makersuite.google.com/app/apikey
        """)
        return None

def main():
    st.set_page_config(
        page_title="Document-Aware AI Assistant with Gemini",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Document-Aware AI Assistant")
   
    st.markdown("Upload a document and interact with it through intelligent Q&A or challenge mode!")
    
    # Get API Key from secrets
    api_key = get_api_key()
    
    if not api_key:
        return
    
    # Initialize session state
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'challenge_questions' not in st.session_state:
        st.session_state.challenge_questions = []
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = None
    if 'document_summary' not in st.session_state:
        st.session_state.document_summary = ""
    
    # Document Upload Section
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=['pdf', 'txt'],
        help="Upload a structured English document (research paper, report, etc.)"
    )
    
    if uploaded_file is not None:
        processor = DocumentProcessor()
        
        with st.spinner("Processing document with Gemini AI..."):
            # Extract text based on file type
            if uploaded_file.type == "application/pdf":
                text = processor.extract_text_from_pdf(uploaded_file)
            else:
                text = processor.extract_text_from_txt(uploaded_file)
            
            if text:
                # Process the document
                processed_data = processor.preprocess_text(text)
                
                # Initialize Gemini assistant
                assistant = GeminiDocumentAssistant(processed_data, api_key)
                
                # Generate summary
                summary = assistant.generate_summary()
                
                # Store in session state
                st.session_state.processed_data = processed_data
                st.session_state.assistant = assistant
                st.session_state.document_processed = True
                st.session_state.document_summary = summary
                
                # Display document info and summary
                st.success("✅ Document processed successfully with Gemini AI!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Word Count", processed_data['word_count'])
                with col2:
                    st.metric("Characters", processed_data['char_count'])
                with col3:
                    st.metric("Paragraphs", len(processed_data['paragraphs']))
                
                st.subheader("📝 AI-Generated Summary (≤150 words)")
                st.info(summary)
    
    # Interaction Modes (only show if document is processed)
    if st.session_state.document_processed and st.session_state.assistant:
        st.header("🤖 Interaction Modes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💬 Ask Anything", use_container_width=True):
                st.session_state.current_mode = "ask_anything"
        
        with col2:
            if st.button("🎯 Challenge Me", use_container_width=True):
                st.session_state.current_mode = "challenge_me"
                with st.spinner("Generating intelligent questions..."):
                    st.session_state.challenge_questions = st.session_state.assistant.generate_challenge_questions()
    
    # Ask Anything Mode
    if st.session_state.current_mode == "ask_anything" and st.session_state.assistant:
        st.subheader("💬 Ask Anything Mode")
        st.write("Ask any question about the uploaded document - Gemini will provide intelligent answers:")
        
        question = st.text_input("Your question:", key="ask_question", placeholder="e.g., What are the main conclusions of this document?")
        
        if st.button("🔍 Get AI Answer", key="ask_submit"):
            if question:
                with st.spinner("Gemini is analyzing the document and generating answer..."):
                    result = st.session_state.assistant.answer_question(question)
                    
                    st.write("**🤖 Gemini's Answer:**")
                    st.success(result['answer'])
                    
                    st.write("**📍 Justification:**")
                    st.info(result['justification'])
                    
                    if result['source_snippet']:
                        with st.expander("📖 Source Content from Document"):
                            st.write(result['source_snippet'])
            else:
                st.warning("Please enter a question!")
        
        # Show conversation history
        if st.session_state.assistant.conversation_history:
            with st.expander("💭 Conversation History"):
                for i, entry in enumerate(reversed(st.session_state.assistant.conversation_history)):
                    st.write(f"**Q{len(st.session_state.assistant.conversation_history)-i} ({entry['timestamp']}):** {entry['question']}")
                    st.write(f"**A:** {entry['answer']}")
                    if entry['source_snippet']:
                        st.caption(f"*Source: {entry['source_snippet'][:100]}...*")
                    st.divider()
    
    # Challenge Me Mode
    elif st.session_state.current_mode == "challenge_me" and st.session_state.assistant:
        st.subheader("🎯 Challenge Me Mode")
        st.write("Answer these AI-generated questions that test your understanding of the document:")
        
        if st.session_state.challenge_questions:
            for i, q_data in enumerate(st.session_state.challenge_questions):
                st.write(f"**Question {i+1} ({q_data.get('type', 'comprehension').title()}):**")
                st.write(q_data['question'])
                st.caption(f"💡 *Hint: {q_data['answer_guidance']}*")
                
                user_answer = st.text_area(
                    f"Your answer for question {i+1}:",
                    key=f"challenge_answer_{i}",
                    height=100,
                    placeholder="Provide a detailed answer based on the document content..."
                )
                
                if st.button(f"📊 Get AI Evaluation", key=f"submit_{i}"):
                    if user_answer:
                        with st.spinner("Gemini is evaluating your answer..."):
                            evaluation = st.session_state.assistant.evaluate_answer(
                                q_data['question'], 
                                user_answer, 
                                q_data.get('type', 'comprehension')
                            )
                            
                            st.write(f"**📊 Score:** {evaluation['score']}")
                            
                            if evaluation['score'] != 'N/A' and '%' in evaluation['score']:
                                try:
                                    score_val = float(evaluation['score'].replace('%', ''))
                                    if score_val >= 80:
                                        st.success(f"🎉 {evaluation['feedback']}")
                                    elif score_val >= 60:
                                        st.warning(f"👍 {evaluation['feedback']}")
                                    else:
                                        st.error(f"📝 {evaluation['feedback']}")
                                except:
                                    st.info(evaluation['feedback'])
                            else:
                                st.info(evaluation['feedback'])
                            
                            st.write(f"**🔍 AI Analysis:** {evaluation['justification']}")
                            
                            if evaluation.get('reference_content'):
                                with st.expander("📖 Relevant Document Content"):
                                    st.write(evaluation['reference_content'])
                    else:
                        st.warning(f"Please enter an answer for question {i+1}!")
                
                st.divider()
        
        if st.button("🔄 Generate New AI Questions"):
            with st.spinner("Gemini is creating new challenge questions..."):
                st.session_state.challenge_questions = st.session_state.assistant.generate_challenge_questions()
                st.rerun()
    
    # Instructions
    if not st.session_state.document_processed:
        st.markdown("""
        ### 📋 How to Use This AI Assistant:
        
        1. **📄 Upload**: Choose a PDF or TXT document
        2. **📊 Review**: Check the AI-generated summary
        3. **🎯 Interact**: Choose your preferred mode:
           - **💬 Ask Anything**: Free-form Q&A powered by Gemini
           - **🎯 Challenge Me**: AI-generated comprehension tests
        4. **🧠 Learn**: Get intelligent answers with detailed justifications
        """)
    
    
if __name__ == "__main__":
    main()
