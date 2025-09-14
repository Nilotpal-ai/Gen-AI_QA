import streamlit as st
import os
import re
import json
import shutil
import tempfile
import cv2
import numpy as np
import fitz
import pytesseract
import wordninja
from PIL import Image
from rapidfuzz import fuzz

from langchain_community.document_loaders.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from qdrant_client.models import VectorParams, Distance
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ------------------ Configuration ------------------
st.set_page_config(page_title="Motor Insurance QA", page_icon="📄")
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pdf_path = "C:/Users/NILOTPAL/OneDrive/Desktop/Motor_Insurance_Handbook.pdf"  # Adjust path as needed
output_folder = "output_pages"
text_output_file = os.path.join(output_folder, "extracted_text.txt")

# ------------------ Helper Functions ------------------

def preprocess_image(image):
    """
    Enhance image for OCR by converting to grayscale,
    adjusting contrast, denoising, and applying adaptive thresholding.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
    denoised = cv2.bilateralFilter(contrast, d=9, sigmaColor=75, sigmaSpace=75)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def extract_text_from_image(image):
    """Extract text from an image using OCR after preprocessing."""
    preprocessed = preprocess_image(image)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(preprocessed, config=custom_config)
    return text

def segment_joined_words(text):
    """
    Split tokens that are accidentally joined using wordninja.
    Only process tokens that are purely alphabetical and longer than a threshold.
    """
    tokens = text.split()
    new_tokens = []
    for token in tokens:
        if token.isalpha() and len(token) > 6:
            splitted = wordninja.split(token)
            if len(splitted) > 1:
                new_tokens.extend(splitted)
            else:
                new_tokens.append(token)
        else:
            new_tokens.append(token)
    return ' '.join(new_tokens)

def clean_extracted_text(text):
    """
    Clean OCR extracted text by removing artifacts, stray punctuation,
    standalone numbers, and fixing malformed words.
    """
    # Fix common concatenated words
    text = re.sub(r'\b(Whatis)', 'What is', text, flags=re.IGNORECASE)
    # Standardize quotes
    text = text.replace(r'\"', '"')
    text = re.sub(r'\s*"\s*', '"', text)
    
    meaningless_words = [
        'pong', 'pang', 'emnisaal', 'emisaal', 'bemtisaal',
        'bemisaa', 'bemisaal', 'bima', 'indies', 'pongal', 'pangal',
        'emn', 'isaal', 'bemi', 'Song'
    ]
    word_pattern = r'\b(?:' + '|'.join([fr'{word[:-1]}[\w]*{word[-1:]}' for word in meaningless_words]) + r')\b'
    text = re.sub(word_pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<=[A-Za-z])\.(?=[A-Za-z])', '', text)
    text = re.sub(r'\.{2,}', ' ', text)
    text = re.sub(r'_{2,}', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\bLT\b', 'I', text)
    text = text.strip()
    text = segment_joined_words(text)
    return text

# ------------------ Red Box Detection Helpers ------------------

LOWER_RED1 = np.array([0, 100, 50])
UPPER_RED1 = np.array([15, 255, 255])
LOWER_RED2 = np.array([160, 100, 50])
UPPER_RED2 = np.array([180, 255, 255])
MIN_BOX_SIZE = 20  # Minimum valid box width/height

def detect_red_boxes(image):
    """Detect red regions using HSV masks and morphology."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    combined_mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def merge_close_boxes(boxes, threshold=20):
    """Merge boxes that are close to each other."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
    merged = [boxes[0]]
    for box in boxes[1:]:
        last = merged[-1]
        if (abs(box[0] - last[2]) < threshold) and (abs(box[1] - last[1]) < threshold):
            merged[-1] = (
                min(last[0], box[0]),
                min(last[1], box[1]),
                max(last[2], box[2]),
                max(last[3], box[3])
            )
        else:
            merged.append(box)
    return merged

def normalize_question(question):
    """
    Normalize a question by standardizing quotes, trimming spaces,
    removing trailing numbers, and fixing missing spaces.
    """
    norm = question.replace(r'\"', '"')
    norm = re.sub(r'\s*"\s*', '"', norm)
    norm = norm.strip()
    norm = re.sub(r'\s*\d+\s*$', '', norm)
    norm = re.sub(r'^(Whatis)', 'What is', norm, flags=re.IGNORECASE)
    return norm

def extract_and_clean_text(pdf_path: str, output_text_file: str):
    """
    Extract text from the PDF using a hybrid approach:
      - Use red-box detection with OCR for highlighted sections.
      - Use native PDF text extraction.
      - Use image OCR for embedded images.
    """
    doc = fitz.open(pdf_path)
    all_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = ""
        # Render high resolution image for red-box extraction
        pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        # Detect red boxes and extract text from those regions
        contours = detect_red_boxes(img)
        boxes = [cv2.boundingRect(cnt) for cnt in contours]
        boxes = [(x, y, x + w, y + h) for (x, y, w, h) in boxes if w > MIN_BOX_SIZE and h > MIN_BOX_SIZE]
        merged_boxes = merge_close_boxes(boxes)
        for box in merged_boxes:
            x0, y0, x1, y1 = box
            cropped = img[y0:y1, x0:x1]
            pil_image = Image.fromarray(cropped)
            extracted_text = extract_text_from_image(np.array(pil_image))
            cleaned_text = clean_extracted_text(extracted_text)
            if cleaned_text:
                page_text += cleaned_text + "\n\n"
        
        # Fallback native text extraction
        native_text = page.get_text("text")
        if native_text.strip():
            page_text += native_text + "\n\n"
        
        # OCR extraction for images embedded in the page
        img_list = page.get_images(full=True)
        for img_index, img_info in enumerate(img_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as img_file:
                    img_file.write(image_bytes)
                    temp_img_path = img_file.name
                ocr_text = extract_text_from_image(cv2.imread(temp_img_path))
                page_text += "\n" + ocr_text
                os.unlink(temp_img_path)
            except Exception as e:
                print(f"Error processing image on page {page_num+1}: {str(e)}")
                continue

        all_text.append(page_text)
    
    raw_text = "\n".join(all_text)
    clean_text = clean_extracted_text(raw_text)
    
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(clean_text)
    print(f"Extracted and cleaned text saved to {output_text_file}")
    return raw_text

def merge_qa_pairs(qa_pairs):
    """
    Merge similar question-answer pairs using fuzzy matching.
    """
    deduped = {}
    for q_raw, a_raw in qa_pairs:
        question = normalize_question(q_raw)
        answer = re.sub(r'\s+', ' ', a_raw.strip())
        merged = False
        for existing_q in list(deduped.keys()):
            score = fuzz.token_set_ratio(existing_q, question)
            if score > 90:
                existing_answer = deduped[existing_q]
                answer_similarity = fuzz.token_set_ratio(existing_answer, answer)
                if answer_similarity < 90:
                    if existing_answer and existing_answer[-1] not in ".?!":
                        deduped[existing_q] = existing_answer.rstrip() + " " + answer
                    else:
                        deduped[existing_q] = answer if len(answer) > len(existing_answer) else existing_answer
                merged = True
                break
        if not merged:
            deduped[question] = answer
    return deduped

def extract_nested_qa_pairs(text_content: str):
    """
    Parse the full text and extract a nested JSON structure of QA pairs or content.
    Sections starting with a number and a dot (e.g., "1. About this handbook") are used for segmentation.
    For FAQ sections, Q/A pairs are extracted.
    """
    nested_qa = {}
    section_pattern = r'(?m)^\s*(\d+)\.\s*(.+)'
    sections = list(re.finditer(section_pattern, text_content))
    num_sections = len(sections)
    
    for i, sec in enumerate(sections):
        section_title = sec.group(2).strip()
        start_index = sec.end()
        end_index = sections[i+1].start() if i < num_sections - 1 else len(text_content)
        section_text = text_content[start_index:end_index].strip()
        
        section_dict = {}
        # If section title indicates FAQ, then extract Q/A pairs
        if "faq" in section_title.lower():
            qa_pattern = r'(?si)Q[.:]\W*([^?]+\?)\s*.*?Ans[.:]\s*([\s\S]*?)(?=(?:\s*Q[.:]|\Z))'
            qa_pairs = re.findall(qa_pattern, section_text)
            deduped = merge_qa_pairs(qa_pairs)
            if deduped:
                for q, a in deduped.items():
                    section_dict[q] = a
            else:
                section_dict["text"] = re.sub(r'\s+', ' ', section_text)
        else:
            # Extract bullet points if available
            bullet_pattern = r'(?m)^•\s*([^:\n]+:)\s*((?:.*(?:\n(?!•).*)*))'
            bullets = re.findall(bullet_pattern, section_text)
            if bullets:
                for bullet in bullets:
                    subheader = bullet[0].strip()
                    content = bullet[1].strip()
                    content = re.sub(rf'^{re.escape(subheader)}\s*', '', content, count=1)
                    lines = section_text.splitlines()
                    bullet_found = False
                    accumulated = content
                    for line in lines:
                        if line.strip().startswith("•"):
                            if subheader in line:
                                bullet_found = True
                                continue
                            elif bullet_found:
                                break
                        elif bullet_found:
                            accumulated += " " + line.strip()
                    section_dict[subheader] = re.sub(r'\s+', ' ', accumulated)
            else:
                section_dict["text"] = re.sub(r'\s+', ' ', section_text)
        
        nested_qa[section_title] = section_dict

    return nested_qa

# ------------------ Hybrid QA System Class ------------------

class HybridQASystem:
    def __init__(self):
        self.qa_nested = {}
        self.flattened_keys = {}  # Map flattened keys to (section, subheader)
        self.setup_qdrant()

    def load_nested_qa_from_dict(self, qa_dict: dict):
        """
        Load nested QA pairs and create a flattened mapping for query matching.
        """
        self.qa_nested = qa_dict
        self.flattened_keys = {}
        for section, subdict in qa_dict.items():
            for subheader, content in subdict.items():
                combined_key = f"{section} :: {subheader}"
                self.flattened_keys[combined_key.lower()] = (section, subheader)

    def query(self, query: str):
        query_lower = query.strip().lower()
        candidates = list(self.flattened_keys.keys())
        best_score = 0
        best_key = None
        for key in candidates:
            score = fuzz.token_set_ratio(query_lower, key)
            if score > best_score:
                best_score = score
                best_key = key
        if best_score > 80:
            section, subheader = self.flattened_keys[best_key]
            answer = self.qa_nested[section][subheader]
            return f"Nested QA Match ({best_score}% confidence):\nSection: {section}\n{subheader}\nAnswer: {answer}"

        # Fallback to semantic search
        try:
            semantic_result = self.qa_chain.invoke({"query": query})
            source_docs = semantic_result.get('source_documents', [])
            answer = semantic_result.get('result', 'No answer found.')

            if source_docs:
                # Safely access scores, default to None if missing
                scores = [doc.metadata.get('score', 'N/A') for doc in source_docs]
                print(f"Retrieved documents with scores: {scores}")
            
            if not source_docs or "No answer found." in answer:
                return "<div class='error'>No answer found.</div>"
            return f"Comprehensive Answer:\n{answer}"
        except Exception as e:
            print(f"Semantic search error: {e}")
            return f"<div class='error'>No answer found. Error: {str(e)}</div>"

    def setup_qdrant(self):
        """
        Setup Qdrant collection and load document embeddings using LangChain components.
        """
        # No API token needed for local models
        loader = TextLoader(text_output_file, encoding="utf-8")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n## ", "\n• ", "\nQ. ", "\n\n", "\n", " "]
        )
        chunks = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.client = QdrantClient("http://localhost:6333")
        self.collection_name = "motor_insurance_v2"

        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Deleted old collection '{self.collection_name}' to recreate with correct dimensions.")
        except Exception as e:
            print(f"Could not delete collection: {e}")

        # Create new collection with correct vector size
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        self.qdrant = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=embeddings
        )

        # Add document chunks to the vector store
        self.qdrant.add_documents(chunks)

        self.retriever = self.qdrant.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.3}
        )

        # Use a free model from Hugging Face that doesn't require special access
        model_name = "microsoft/DialoGPT-medium"
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2
        )
        
        # Create LLM from pipeline
        self.llm = HuggingFacePipeline(pipeline=pipe)

        prompt_template = PromptTemplate(
            template=( 
                "Answer using ONLY the context from the motor insurance handbook.\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "If the context doesn't explicitly contain the answer, say 'No answer found'.\n"
                "Answer:"
            ),
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )

# ------------------ Streamlit Cached Loader ------------------

@st.cache_resource
def load_qa_system():
    # Clear (or create) the output folder
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Process the PDF and load the QA system
    raw_text = extract_and_clean_text(pdf_path, text_output_file)
    nested_qa_dict = extract_nested_qa_pairs(raw_text)
    
    system = HybridQASystem()
    system.load_nested_qa_from_dict(nested_qa_dict)
    print(f"Loaded Nested QA pairs: {len(system.flattened_keys)} keys")
    return system

# ------------------ Streamlit UI ------------------

def main():
    # Custom CSS for a unique, modern, interactive look with a one-time slide animation
    st.markdown("""
    <style>
    /* General styling */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stTextInput input {
        border: 2px solid #2a76be !important;
        border-radius: 5px !important;
        padding: 8px;
    }
    .stAlert, .stWarning, .stError {
        border-left: 4px solid #2a76be !important;
    }
    /* Header styling */
    .main-header {
        color: #2a76be;
        border-bottom: 2px solid #2a76be;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    /* One-time slide-in animation for the animated-text class */
    @keyframes slideIn {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    .animated-text {
        display: inline-block;
        font-size: 16px;
        font-weight: 500;
        /* You can keep the gradient or change to a solid color as desired */
        background: linear-gradient(-45deg, #ff4e50, #f9d423, #1fa2ff, #12d8fa);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: slideIn 3s ease-out forwards;
    }
    /* Styling for no answer found animation */
    .error {
        color: #c0392b;
        font-weight: bold;
        animation: shake 0.5s;
    }
    @keyframes shake {
        0% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        50% { transform: translateX(5px); }
        75% { transform: translateX(-5px); }
        100% { transform: translateX(0); }
    }
    /* Comprehensive answer styling */
    .comp-answer {
        background: #f0faf0;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    /* Nested QA match styling */
    .nested-answer {
        background: #e8f4ff;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <h1>🚗 Motor Insurance QA Portal</h1>
        <p class="animated-text">Ask questions about insurance policies, claims, and regulations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize the QA system (this may take a moment on first run)
    try:
        qa_system = load_qa_system()
    except Exception as e:
        st.error(f"Error loading QA system: {str(e)}")
        st.info("Please ensure Qdrant is running on localhost:6333")
        return

    # User query input
    query = st.text_input("Enter your question:", placeholder="e.g. What is No Claim Bonus?", key="query_input")

    if query:
        with st.spinner("Searching for answers..."):
            try:
                response = qa_system.query(query)
                
                # Convert URLs to clickable hyperlinks first
                response = response.replace("www.irda.gov.in", 
                    "<a href='https://www.irda.gov.in' target='_blank' style='color: #2a76be;'>www.irda.gov.in</a>")
                
                # Check and display Nested QA matches
                if "Nested QA Match" in response:
                    parts = response.split("\n")
                    if len(parts) >= 4:
                        st.markdown(f"""
                        <div class='nested-answer'>
                            <h3 style='color: #2a76be; margin-top: 0;'>{parts[2]}</h3>
                            <p>{parts[3]}</p>
                            <small>{parts[0]}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info(response)
                # Display comprehensive answer with dedicated style
                elif "Comprehensive Answer" in response:
                    answer_text = response.split(":", 1)[1].strip()
                    st.markdown(f"""
                    <div class='comp-answer'>
                        ✅ <strong>Comprehensive Answer:</strong><br>
                        {answer_text}
                    </div>
                    """, unsafe_allow_html=True)
                # For any errors or no answer found, show the animated error div
                else:
                    st.markdown(response, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()