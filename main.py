"""
Complete Local RAG System - Quiz Generator & Tutor
Single file, robust, accurate with proper citations
"""

import streamlit as st #UI
import chromadb   #Vector DB
import fitz #it is a PyMuPDF module
import html
import os
import json
from pathlib import Path #it is used for handling file paths
from ddgs import DDGS #It is used for DuckDuckGo searches
import requests #It is used for making HTTP requests
import wikipedia #it is used for Wikipedia searches
import streamlit.components.v1 as components


# Configuration - Directory and API settings
ROOT = Path(__file__).parent
SOURCE_DIR = "Source"  # Directory containing source PDF files
VECTOR_DB_DIR = "./vectordb"  # Directory for ChromaDB vector database storage
PROCESSED_DIR = "./processed_md"  # Directory for processed markdown files
COLLECTION_NAME = "knowledge_base"  # Name of the ChromaDB collection
OLLAMA_URL = "http://localhost:11434"  # Local Ollama API endpoint

PROJECT_DOCS = {
    "Architecture notes": ROOT / "ARCHITECTURE.md",
    "Repository README": ROOT / "README.md",
}
DIAGRAM_PATH = ROOT / "architecture-diagram.mmd"


def render_mermaid_diagram(diagram_text):
    """Render Mermaid text in Streamlit while keeping the source available."""
    escaped_diagram = html.escape(diagram_text)
    html_doc = f"""
    <pre class="mermaid">{escaped_diagram}</pre>
    <script type="module">
      import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
      mermaid.initialize({{ startOnLoad: true, securityLevel: "loose" }});
    </script>
    """
    components.html(html_doc, height=900, scrolling=True)


# ============================================================================
# PDF Processing
# ============================================================================

def process_pdfs():
    """Convert PDFs to markdown and create vector embeddings
    
    This function performs the complete PDF processing pipeline:
    1. Creates necessary directories
    2. Loads all PDFs from SOURCE_DIR
    3. Extracts text page-by-page using PyMuPDF
    4. Saves extracted text as markdown files
    5. Creates document chunks with metadata
    6. Generates embeddings and stores in ChromaDB
    """
    
    # Create directories if they don't exist (exist_ok=True prevents errors if already present)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    
    # Get PDF files using glob pattern to find all .pdf files in SOURCE_DIR
    pdf_files = list(Path(SOURCE_DIR).glob("*.pdf"))
    if not pdf_files:
        st.error(f"No PDFs found in {SOURCE_DIR}")
        return None
    
    st.info(f"Found {len(pdf_files)} PDF files")
    
    # Initialize ChromaDB with persistent storage (data saved to disk)
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    
    # Recreate collection to ensure fresh indexing (delete old if exists)
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass  # Ignore error if collection doesn't exist
    
    # Create new collection with cosine similarity for semantic search
    # HNSW (Hierarchical Navigable Small World) is an efficient similarity search algorithm
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Cosine similarity measures angle between vectors
    )
    
    # Process each PDF - iterate through all PDF files and extract content
    all_chunks = []  # Store all document chunks for batch processing
    progress_bar = st.progress(0)  # Visual progress indicator in Streamlit UI
    
    # Iterate through each PDF file with enumeration for progress tracking
    for idx, pdf_path in enumerate(pdf_files): 
        st.text(f"Processing: {pdf_path.name}")
        
        # Extract text from PDF using PyMuPDF (fitz)
        pdf = fitz.open(pdf_path)  # Open PDF document
        doc_name = pdf_path.stem  # Get document name without .pdf extension
        
        # Initialize markdown content with document title
        md_content = f"# {doc_name}\n\n"
        
        # Process each page in the PDF document
        for page_num in range(len(pdf)):
            page = pdf[page_num]  # Get page object
            text = page.get_text()  # Extract all text from the page
            
            if text.strip(): # Only process non-empty pages #Strip is used to remove leading and trailing whitespace
                md_content += f"## Page {page_num + 1}\n\n{text}\n\n"
                
                # Create chunk consists of page text and metadata. It will be used for vector DB for indexing and getting context and retrieval
                all_chunks.append({
                    'id': f"{doc_name}_p{page_num + 1}",
                    'text': text,
                    'source': doc_name,
                    'page': page_num + 1
                })
        
        pdf.close()
        
        # The page numbers and metadata are added to all_chunks list
        # Save markdown in processed directory for human-readable backup
        md_path = Path(PROCESSED_DIR) / f"{doc_name}.md"  # Construct file path
        with open(md_path, 'w', encoding='utf-8') as f:  # UTF-8 encoding handles special characters
            f.write(md_content)
        
        # Update progress bar: (current file index + 1) / total files
        progress_bar.progress((idx + 1) / len(pdf_files))
    
    # Add all chunks to vector database with automatic embedding generation
    st.info("Creating embeddings...")
    
    # Process chunks in batches to avoid memory issues with large document sets
    # The flow: extract text, metadata, and IDs from batch → ChromaDB generates embeddings → store in database
    batch_size = 100  # Process 100 chunks at a time
    for i in range(0, len(all_chunks), batch_size):  # Iterate in steps of batch_size
        batch = all_chunks[i:i + batch_size]  # Get current batch slice
        
        # Add batch to collection - ChromaDB automatically generates embeddings for documents
        collection.add(
            documents=[c['text'] for c in batch],  # List comprehension to extract text
            metadatas=[{'source': c['source'], 'page': c['page']} for c in batch],  # Extract metadata
            ids=[c['id'] for c in batch]  # Unique identifiers for each chunk
        )
    
    st.success(f"Processed {len(pdf_files)} PDFs, {len(all_chunks)} pages indexed")
    return collection


# ============================================================================
# Search & Retrieval
# ============================================================================

# The following function searches the local knowledge base using the provided query and returns the top_k results along with their metadata and scores.
# top_k is the number of top results to return
# We set value to 5 because we want to retrieve the top 5 most relevant documents from the local knowledge base for a given query.

def search_local(query, collection, top_k=5):
    """Search local knowledge base using semantic similarity
    
    Args:
        query: Search query string
        collection: ChromaDB collection object
        top_k: Number of most relevant results to return (default: 5)
    
    Returns:
        List of dictionaries containing text, source, page, and relevance score
    """
    try:
        # Query ChromaDB - it automatically converts query to embedding and finds nearest neighbors
        results = collection.query(
            query_texts=[query],  # Query text (automatically embedded by ChromaDB)
            n_results=top_k  # Number of results to retrieve
        )
        
        # Check if any results were found
        if not results['documents'] or not results['documents'][0]:
            return []  # Return empty list if no results
        
        # Transform ChromaDB results into structured format
        retrieved = []
        for i in range(len(results['documents'][0])):
            retrieved.append({
                'text': results['documents'][0][i],  # Document text content
                'source': results['metadatas'][0][i]['source'],  # Source document name
                'page': results['metadatas'][0][i]['page'],  # Page number in the original PDF document
                'score': 1 - results['distances'][0][i]  # Convert distance to similarity score (higher = more similar)
            })
        
        return retrieved
    except:
        return []


# If the information is not found locally, the following function searches Wikipedia for relevant information based on the provided query.
def search_wikipedia(query):
    """Search Wikipedia for information as a fallback source
    
    Args:
        query: Search query string
    
    Returns:
        List of dictionaries with Wikipedia article content
    """
    try:
        wikipedia.set_lang("en")  # Set language to English
        
        # Search for Wikipedia pages matching the query
        search_results = wikipedia.search(query, results=3)  # Get top 3 matching articles
        
        if not search_results:
            return []
        
        wiki_results = []
        # Iterate through search results and fetch full page content
        for title in search_results[:3]:  # Process up to 3 articles
            try:
                page = wikipedia.page(title, auto_suggest=False)  # Fetch page without auto-correction
                
                # Get first 2000 characters of article content as summary
                summary = page.content[:2000]
                
                wiki_results.append({
                    'text': f"{page.title}\n\n{summary}",
                    'source': page.url,
                    'page': 'Wikipedia',
                    'score': 0.9
                })
            except:
                continue
        
        return wiki_results
    except Exception as e:
        return []

# The following function performs a comprehensive web search using DuckDuckGo. If DuckDuckGo fails to return results, it falls back to searching Wikipedia.

def search_web(query):
    """Comprehensive web search - fetches from internet using DuckDuckGo
    
    Args:
        query: Search query string
    
    Returns:
        List of search results from web or Wikipedia
    """
    
    web_results = []
    
    # Try DuckDuckGo text search as primary web search method
    try:
        st.info("Searching the internet...")
        with DDGS() as ddgs:  # Context manager handles connection cleanup
            results = list(ddgs.text(query, max_results=10))  # Fetch top 10 search results
            
            if results:
                for r in results:
                    content = f"{r.get('title', '')}\n\n{r.get('body', '')}"
                    web_results.append({
                        'text': content,
                        'source': r.get('href', 'Unknown'),
                        'page': 'Web Search',
                        'score': 0.85
                    })
                
                if web_results:
                    st.success(f"Found {len(web_results)} results from internet")
                    return web_results
    except Exception as e:
        st.warning(f"DuckDuckGo search error: {str(e)[:100]}")
    
    # If DuckDuckGo fails, try Wikipedia as last resort
    st.info("Trying Wikipedia as fallback...")
    wiki_results = search_wikipedia(query)
    
    if wiki_results:
        st.success(f"Found {len(wiki_results)} results from Wikipedia")
        return wiki_results
    
    st.error("All internet search methods failed")
    return []


# The following function attempts to retrieve context for a given query. It first searches the local knowledge base, and if the results do not meet a specified relevance threshold, it performs a web search.
# Threshold is a relevance score threshold to decide if local results are sufficient.
def get_context(query, collection, threshold=0.4):
    """Get context from local knowledge base or web
    
    Strategy:
    1. Search local ChromaDB first
    2. If top result score > threshold, use local results
    3. Otherwise, fall back to web search
    
    Args:
        query: Search query string
        collection: ChromaDB collection
        threshold: Minimum score to accept local results (default: 0.4)
    
    Returns:
        Tuple of (results_list, source_type)
    """
    # Try local search first (fastest and most reliable)
    local = search_local(query, collection)
    
    # Check if local results are good enough (score above threshold)
    if local and len(local) > 0 and local[0]['score'] > threshold:
        return local, "local"  # Return local results with source indicator
    
    # Local search didn't find relevant results, try web search
    st.warning("Topic not in local sources, searching web...")
    web = search_web(query)  # Fallback to DuckDuckGo/Wikipedia
    
    if not web or len(web) == 0:
        st.error("Web search returned no results")
        return [], "none"
    
    return web, "web"


# ============================================================================
# LLM Integration
# ============================================================================
# This section integrates with the local LLM (Ollama) to generate quiz questions based on retrieved context.
# The grading is done by comparing user answers to correct answers and providing detailed feedback.
# The quiz data structure is a list of dictionaries, each representing a question with:
# - type: 'MCQ', 'TF', or 'FIB'
# - question: The question text
# - options: List of answer choices (empty for FIB)
# - correct: The correct answer
# - explanation: Detailed explanation with source citation
# - source: Source document name
# - page: Page number in source

def check_ollama(show_messages=True):
    """Check if Ollama server is running and accessible
    
    Returns:
        Boolean indicating if Ollama is available
    """
    try:
        # Ping Ollama API to check if service is running
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            return True
        if show_messages:
            st.warning(f"Ollama returned status {response.status_code}")
        return False
    except Exception as e:
        if show_messages:
            st.error(f"Cannot connect to Ollama: {str(e)[:100]}")
            st.info("Start Ollama by running: **ollama serve**")
        return False


def generate_with_ollama(prompt, model="qwen2.5:7b-instruct"):
    """Generate response with Ollama LLM
    
    Args:
        prompt: The prompt text to send to the model
        model: Model name (default: qwen2.5:7b-instruct)
    
    Returns:
        Generated text response or None if failed
    """
    try:
        # Send POST request to Ollama generate endpoint
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,  # Which LLM model to use
                "prompt": prompt,  # The input prompt
                "stream": False,  # Get complete response (not streaming)
                "options": {
                    "temperature": 0.7,  # Creativity level (0=deterministic, 1=creative)
                    "num_predict": 2000  # Maximum tokens to generate
                }
            },
            timeout=120  # Wait up to 2 minutes for response
        )
        if response.status_code == 200:
            return response.json()['response']
        else:
            st.error(f"Ollama returned status {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Ollama error: {e}")
        st.info("Run 'ollama serve' in a terminal to start Ollama")
        return None


# ============================================================================
# Quiz Generation
# ============================================================================

def generate_quiz(topic, context, num_q=5):
    """Generate quiz with LLM - includes MCQ, True/False, and Fill-in-the-blank
    
    This function:
    1. Prepares context from retrieved documents
    2. Constructs detailed prompt with instructions and examples
    3. Calls Ollama LLM to generate questions
    4. Parses and validates the response
    5. Retries if necessary to get all required questions
    
    Args:
        topic: The quiz topic
        context: List of retrieved context documents
        num_q: Number of questions to generate (default: 5)
    
    Returns:
        List of question dictionaries or None if generation fails
    """
    
    # Prepare context text from top 5 retrieved documents (first 1000 chars each)
    context_text = "\n\n".join([
        f"[Source: {c['source']}, Page {c['page']}]\n{c['text'][:1000]}"
        for c in context[:5]
    ])
    
    # Calculate question type distribution to ensure variety
    num_mcq = max(1, num_q - 3)  # At least 1 MCQ (multiple choice question)
    num_tf = 2  # Exactly 2 True/False questions
    num_fib = 1  # Exactly 1 Fill-in-the-blank question
    # For num_q=5: 2 MCQ + 2 TF + 1 FIB = 5 total
    
    prompt = f"""You are an expert quiz generator. Create a high-quality quiz about "{topic}" using ONLY the information from the context below. Do NOT make up information.

CONTEXT:
{context_text}

CRITICAL RULES:
1. Extract SPECIFIC facts, definitions, and technical details from the context
2. Use EXACT terminology and numbers from the sources
3. Create CLEAR, UNAMBIGUOUS questions WITHOUT mentioning source names in the question text
4. Make wrong options plausible but clearly incorrect
5. For True/False: use complete factual statements WITHOUT source references in the statement
6. For Fill-in-the-blank: create natural sentences WITHOUT mentioning sources
7. Keep source references ONLY in explanations, NOT in questions

YOU MUST GENERATE EXACTLY {num_mcq} MULTIPLE CHOICE + {num_tf} TRUE/FALSE + {num_fib} FILL-IN-THE-BLANK = {num_q} TOTAL QUESTIONS.

FORMAT FOR MULTIPLE CHOICE:
QUESTION [number]: MCQ
What is [specific technical aspect] of {topic}? [DO NOT mention sources here]
A) [Correct answer with specific details from context]
B) [Plausible wrong answer]
C) [Plausible wrong answer]
D) [Plausible wrong answer]
CORRECT: A
EXPLANATION: According to [Source], page [X]: [Quote the relevant sentence]

---

FORMAT FOR TRUE/FALSE:
QUESTION [number]: TF
[Complete factual statement - NO source mentions in the statement itself]
A) True
B) False
CORRECT: A
EXPLANATION: This is true according to [Source], page [X]: [Quote supporting text]

---

FORMAT FOR FILL-IN-THE-BLANK:
QUESTION [number]: FIB
[Natural sentence with _____ for the blank - NO source mentions in the sentence]
CORRECT: [The exact term that was removed]
EXPLANATION: Complete sentence from [Source], page [X]: [Full original sentence]

---

EXAMPLE MCQ:
QUESTION 1: MCQ
What is the key size used in AES-128?
A) 128 bits
B) 64 bits
C) 256 bits
D) 192 bits
CORRECT: A
EXPLANATION: According to Network-security-essentials, page 45: AES-128 uses a 128-bit key.

Now generate the quiz with SPECIFIC details from the context:"""
    
    # Check if Ollama service is available
    ollama_available = check_ollama()
    
    if not ollama_available:
        st.error("Ollama is not running. Please start Ollama to generate contextual questions.")
        st.info("Run: **ollama serve** in a terminal")
        return None  # Cannot proceed without LLM
    
    st.info("Generating quiz with local LLM (qwen2.5:7b-instruct)...")
    
    # Retry mechanism: Try up to 2 times to get all required questions
    # LLMs are non-deterministic and may not always generate complete output
    for attempt in range(2):  # Attempt 0 and 1
        # Generate quiz using Ollama LLM
        response = generate_with_ollama(prompt)
        
        # Check if LLM returned a response
        if not response:
            st.error("LLM failed to generate response")
            if attempt == 0:  # First attempt failed
                st.info("Retrying...")
                continue  # Try again
            return None  # Both attempts failed
        
        # Parse LLM response into structured quiz format
        parsed = parse_quiz(response, context, num_q)
        
        # Check if we got enough questions
        if parsed and len(parsed) >= num_q:
            st.success(f"Generated {len(parsed)} contextual questions")
            return parsed[:num_q]  # Return exactly num_q questions
        elif parsed and len(parsed) > 0:
            # Got some questions but not enough
            st.warning(f"LLM only generated {len(parsed)}/{num_q} questions")
            if attempt == 0:  # First attempt
                st.info("Retrying to get all 5 questions...")
                continue  # Try again
            else:  # Second attempt also insufficient
                st.error(f"Could only generate {len(parsed)} questions after 2 attempts")
                return parsed  # Return partial quiz rather than nothing
        else:
            # Parsing completely failed
            st.error("Failed to parse LLM output")
            if attempt == 0:  # First attempt
                st.info("Retrying...")
                continue  # Try again
            return None  # Give up after 2 attempts
    
    return None


def create_template_quiz(topic, context, num_q):
    """Generate high-quality quiz from context content with MCQ, TF, and FIB questions"""
    quiz = []
    
    num_mcq = max(1, num_q - 3)
    num_tf = 2
    num_fib = 1
    
    # Generate MCQ questions - extract specific facts
    for i in range(min(num_mcq, len(context))):
        ctx = context[i]
        text = ctx['text']
        
        # Extract complete sentences with technical content
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 40 and any(char.isupper() for char in s)]
        if len(sentences) < 1:
            continue
        
        # Use the most informative sentence
        main_fact = sentences[0] if len(sentences[0]) < 200 else sentences[0][:197] + "..."
        
        # Extract a key term to make question about
        words = main_fact.split()
        key_terms = [w for w in words if len(w) > 4 and w[0].isupper()]
        topic_word = key_terms[0] if key_terms else topic
        
        # Create specific question without source reference
        question = f"Which statement about {topic} is correct?"
        
        # Create more realistic options
        options = [
            f"A) {main_fact}",
            f"B) {topic} is not mentioned in security literature",
            f"C) {topic} has been deprecated and is no longer used",
            f"D) {topic} is only used in legacy systems"
        ]
        
        quiz.append({
            'type': 'MCQ',
            'question': question,
            'options': options,
            'correct': 'A',
            'explanation': f"Correct. The source states: '{main_fact}' ({ctx['source']}, page {ctx['page']})",
            'source': ctx['source'],
            'page': ctx['page']
        })
    
    # Generate True/False questions - use factual statements
    for i in range(min(num_tf, len(context))):
        ctx_idx = min(i + num_mcq, len(context) - 1)
        ctx = context[ctx_idx]
        sentences = [s.strip() for s in ctx['text'].split('.') if 30 < len(s.strip()) < 150]
        
        if sentences:
            # Pick a clear factual statement
            statement = sentences[0]
            quiz.append({
                'type': 'TF',
                'question': f"{statement}.",
                'options': ['A) True', 'B) False'],
                'correct': 'A',
                'explanation': f"True. This is stated in {ctx['source']}, page {ctx['page']}",
                'source': ctx['source'],
                'page': ctx['page']
            })
    
    # Generate Fill-in-the-blank question - remove technical term
    if len(context) > 0:
        ctx = context[0]
        sentences = [s.strip() for s in ctx['text'].split('.') if 40 < len(s.strip()) < 150]
        if sentences:
            sentence = sentences[0]
            words = sentence.split()
            
            # Find a good word to blank out (technical term or number)
            blank_idx = -1
            blank_word = ""
            
            # Prefer capitalized technical terms or numbers
            for idx, word in enumerate(words):
                if len(word) > 3 and (word[0].isupper() or word.isdigit()):
                    blank_idx = idx
                    blank_word = word
                    break
            
            # Fallback to middle word
            if blank_idx == -1 and len(words) > 5:
                blank_idx = len(words) // 2
                blank_word = words[blank_idx]
            
            if blank_idx >= 0:
                words[blank_idx] = '_____'
                quiz.append({
                    'type': 'FIB',
                    'question': ' '.join(words) + ".",
                    'options': [],
                    'correct': blank_word.strip('.,!?;:()[]'),
                    'explanation': f"Complete sentence: '{sentence}' ({ctx['source']}, page {ctx['page']})",
                    'source': ctx['source'],
                    'page': ctx['page']
                })
    
    return quiz if quiz else [{
        'type': 'MCQ',
        'question': f"What information is available about {topic}?",
        'options': [
            f"A) {context[0]['text'][:150]}...",
            "B) No information found",
            "C) Conflicting information",
            "D) Unclear from sources"
        ],
        'correct': 'A',
        'explanation': f"See {context[0]['source']}, page {context[0]['page']}",
        'source': context[0]['source'],
        'page': context[0]['page']
    }]


def parse_quiz(text, context, num_q):
    """Parse LLM output into structured quiz format with MCQ, TF, and FIB support
    
    The parser expects LLM output in a specific format:
    - Questions marked with "QUESTION [number]:"
    - Question type: MCQ, TF, or FIB
    - Options marked with A), B), C), D)
    - Correct answer marked with "CORRECT:"
    - Explanation marked with "EXPLANATION:"
    
    Args:
        text: Raw text output from LLM
        context: Source documents (for metadata)
        num_q: Expected number of questions
    
    Returns:
        List of parsed question dictionaries or None if parsing fails
    """
    import re
    
    quiz = []  # Store parsed questions
    
    # Split text by question markers using regex
    question_blocks = re.split(r'QUESTION \d+:', text)  # Split on "QUESTION 1:", "QUESTION 2:", etc.
    question_blocks = [q.strip() for q in question_blocks if q.strip()]  # Remove empty blocks
    
    # Process each question block
    for block in question_blocks:
        try:
            # Remove separator if present ("---" is used to separate questions in prompt)
            block = block.split('---')[0].strip()
            
            # Determine question type from block prefix
            q_type = 'MCQ'  # Default to multiple choice
            if block.startswith('TF'):  # True/False question
                q_type = 'TF'
                block = block[2:].strip()  # Remove "TF" prefix
            elif block.startswith('FIB'):  # Fill-in-the-blank question
                q_type = 'FIB'
                block = block[3:].strip()  # Remove "FIB" prefix
            elif block.startswith('MCQ'):  # Multiple choice (explicit)
                block = block[3:].strip()  # Remove "MCQ" prefix
            
            # Parse question components from block
            lines = block.split('\n')  # Split block into lines
            question_text = []  # Store question text lines
            options = []  # Store answer options
            correct = None  # Store correct answer
            explanation = ""  # Store explanation text
            
            i = 0  # Line counter
            # Extract question text (everything before options or CORRECT marker)
            while i < len(lines) and not lines[i].strip().startswith(('A)', 'B)', 'C)', 'D)', 'CORRECT:')):
                if lines[i].strip():  # Skip empty lines
                    question_text.append(lines[i].strip())
                i += 1
            
            # Extract options (for MCQ and TF questions only)
            if q_type in ['MCQ', 'TF']:
                while i < len(lines):
                    line = lines[i].strip()
                    if line.startswith(('A)', 'B)', 'C)', 'D)')):  # Option line
                        options.append(line)
                    elif line.startswith('CORRECT:'):  # Correct answer marker
                        correct = line.split(':', 1)[1].strip().upper()  # Extract letter (A, B, C, or D)
                    elif line.startswith('EXPLANATION:'):  # Explanation marker
                        explanation = line.split(':', 1)[1].strip()  # Get text after colon
                        # Get rest of explanation (may span multiple lines)
                        i += 1
                        while i < len(lines) and not lines[i].strip().startswith('QUESTION'):
                            if lines[i].strip():
                                explanation += " " + lines[i].strip()  # Append continuation
                            i += 1
                        break  # Done with this question
                    i += 1
            else:  # FIB - Fill-in-the-blank has no options, just correct word and explanation
                while i < len(lines):
                    line = lines[i].strip()
                    if line.startswith('CORRECT:'):  # The missing word/phrase
                        correct = line.split(':', 1)[1].strip()  # Extract the answer
                    elif line.startswith('EXPLANATION:'):  # Full sentence explanation
                        explanation = line.split(':', 1)[1].strip()
                        i += 1
                        # Collect multi-line explanation
                        while i < len(lines) and not lines[i].strip().startswith('QUESTION'):
                            if lines[i].strip():
                                explanation += " " + lines[i].strip()
                            i += 1
                        break
                    i += 1
            
            # Validate parsed question components and add to quiz
            if question_text and correct:  # Basic validation: must have question and answer
                # Validate MCQ: needs 4 options and correct answer must be A, B, C, or D
                if q_type == 'MCQ' and len(options) >= 4 and correct in ['A', 'B', 'C', 'D']:
                    quiz.append({
                        'type': 'MCQ',
                        'question': ' '.join(question_text),  # Join multi-line questions
                        'options': options[:4],  # Take first 4 options
                        'correct': correct,  # Letter A-D
                        'explanation': explanation if explanation else "See context above.",  # Fallback explanation
                        'source': context[0]['source'] if context else "Unknown",  # Source document name
                        'page': context[0]['page'] if context else 0  # Page number
                    })
                elif q_type == 'TF' and len(options) >= 2 and correct in ['A', 'B']:
                    quiz.append({
                        'type': 'TF',
                        'question': ' '.join(question_text),
                        'options': options[:2],
                        'correct': correct,
                        'explanation': explanation if explanation else "See context above.",
                        'source': context[0]['source'] if context else "Unknown",
                        'page': context[0]['page'] if context else 0
                    })
                elif q_type == 'FIB' and correct:
                    quiz.append({
                        'type': 'FIB',
                        'question': ' '.join(question_text),
                        'options': [],
                        'correct': correct,
                        'explanation': explanation if explanation else "See context above.",
                        'source': context[0]['source'] if context else "Unknown",
                        'page': context[0]['page'] if context else 0
                    })
        except Exception as e:
            st.warning(f"Failed to parse question: {str(e)[:100]}")
            continue
    
    if len(quiz) == 0:
        st.error("Failed to parse any questions from LLM output")
        return None
    
    return quiz


def grade_answers(quiz, answers):
    """Grade quiz answers with detailed feedback - supports MCQ, TF, and FIB
    
    This function:
    1. Compares user answers against correct answers
    2. Handles different question types (MCQ, TF, FIB) appropriately
    3. Generates detailed feedback with explanations and source citations
    4. Calculates total score
    
    Args:
        quiz: List of question dictionaries
        answers: List of user's answers (same order as quiz)
    
    Returns:
        Tuple of (score, feedback_list)
    """
    score = 0  # Initialize score counter
    feedback = []  # Store feedback for each question
    
    # Iterate through questions and answers simultaneously using zip
    for i, (q, ans) in enumerate(zip(quiz, answers)):
        correct = q['correct']  # Get correct answer from question
        question_num = i + 1  # 1-based numbering for display
        q_type = q.get('type', 'MCQ')  # Get question type (default to MCQ)
        
        # Check if user's answer is correct (different logic for different types)
        is_correct = False
        if q_type == 'FIB':
            # Case-insensitive comparison for fill-in-the-blank (allows minor variations)
            is_correct = ans and ans.strip().lower() == correct.strip().lower()
        else:
            # Exact match for MCQ and TF (just compare letter: A, B, C, or D)
            is_correct = ans == correct
        
        if is_correct:
            score += 1  # Increment score for correct answer
            # Format source citation with document name and page number
            source_ref = f"[{q.get('source', 'Unknown')}, Page {q.get('page', 'N/A')}]"
            # Add positive feedback with explanation and source
            feedback.append(f"**Question {question_num}: CORRECT**\n{q['explanation']}\n*Source: {source_ref}*")
        else:
            # Answer is incorrect - provide detailed feedback
            source_ref = f"[{q.get('source', 'Unknown')}, Page {q.get('page', 'N/A')}]"
            
            if q_type == 'FIB':
                feedback.append(
                    f"**Question {question_num}: INCORRECT**\n"
                    f"Your answer: {ans if ans else '(blank)'}\n"
                    f"Correct answer: {correct}\n"
                    f"{q['explanation']}\n"
                    f"*Source: {source_ref}*"
                )
            else:
                correct_option = [opt for opt in q['options'] if opt.startswith(correct + ')')][0] if q['options'] else correct
                feedback.append(
                    f"**Question {question_num}: INCORRECT**\n"
                    f"Your answer: {ans if ans else '(not answered)'}\n"
                    f"Correct answer: {correct_option}\n"
                    f"{q['explanation']}\n"
                    f"*Source: {source_ref}*"
                )
    
    return score, feedback


# ============================================================================
# Tutor Mode
# ============================================================================

def generate_explanation(topic, context):
    """Generate detailed explanation with citations using Ollama LLM
    
    This function is used in Tutor mode to provide comprehensive explanations
    of concepts based on retrieved context from the knowledge base.
    
    Args:
        topic: The topic/question to explain
        context: List of retrieved context documents
    
    Returns:
        Tuple of (explanation_text, context_list)
    """
    
    # Prepare context with source information (top 5 sources, first 800 chars each)
    context_blocks = []
    for i, c in enumerate(context[:5], 1):  # Use up to 5 sources, numbered 1-5
        context_blocks.append(
            f"[Source {i}] {c['source']}, Page {c['page']}:\n{c['text'][:800]}"
        )
    
    # Join all context blocks with double newlines for readability
    context_text = "\n\n".join(context_blocks)
    
    if check_ollama():  # Check if Ollama is running
        st.info("Generating explanation with local LLM...")
        
        # Construct prompt for tutor mode with detailed instructions
        prompt = f"""You are an expert tutor. Provide a detailed, accurate explanation of: "{topic}"

Use ONLY the information from these sources:

{context_text}

INSTRUCTIONS:
- Explain the concept clearly and thoroughly
- Use specific details from the sources
- Cite sources in your explanation like [Source 1, Page X]
- Organize information logically
- Use examples from the sources when available
- Be accurate - don't add information not in the sources

Provide your explanation:"""
        
        # Generate explanation using Ollama
        response = generate_with_ollama(prompt)
        if response:
            st.success("Explanation generated")
            return response, context  # Return LLM-generated explanation
        else:
            st.warning("LLM generation failed. Showing raw context.")
    
    # Fallback: If Ollama is not available, format context nicely as markdown
    formatted = f"# Information about: {topic}\n\n"
    for i, c in enumerate(context[:5], 1):
        formatted += f"## Source {i}: {c['source']} (Page {c['page']})\n\n"
        formatted += f"{c['text'][:1000]}\n\n"  # Show first 1000 chars of each source
        formatted += "---\n\n"  # Separator between sources
    
    return formatted, context  # Return formatted raw context


# ============================================================================
# Streamlit UI
# ============================================================================

def main():
    """Main Streamlit application entry point
    
    This function:
    1. Configures the Streamlit page
    2. Initializes the vector database connection
    3. Renders the UI based on selected mode (Quiz or Tutor)
    4. Handles user interactions and state management
    """
    st.set_page_config(page_title="Network Security RAG Teaching Bot", layout="wide")

    st.title("Network Security RAG Teaching Bot")
    st.caption(
        "A local AI study assistant that answers course questions, generates quizzes, "
        "and shows document citations from network security materials."
    )

    pdf_count = len(list(Path(SOURCE_DIR).glob("*.pdf")))
    processed_count = len(list(Path(PROCESSED_DIR).glob("*.md")))

    if 'collection' not in st.session_state:
        try:
            client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
            collection = client.get_collection(COLLECTION_NAME)
            st.session_state.collection = collection
        except:
            st.session_state.collection = None

    collection = st.session_state.collection
    doc_count = collection.count() if collection else 0
    ollama_status = check_ollama(show_messages=False)
    source_files = sorted(Path(SOURCE_DIR).glob("*.pdf"))

    with st.sidebar:
        st.header("System Status")
        st.metric("Indexed chunks", doc_count)
        st.metric("Source PDFs", pdf_count)
        st.metric("Processed notes", processed_count)
        st.info(f"Ollama: {'Running' if ollama_status else 'Not running'}")
        if not ollama_status:
            st.caption("For generated answers, run `ollama serve` and pull `qwen2.5:7b-instruct`.")

        if st.button("Process PDFs", type="primary", use_container_width=True):
            with st.spinner("Indexing PDFs into the local vector database..."):
                collection = process_pdfs()
                if collection:
                    st.session_state.collection = collection
                    st.rerun()

    overview, architecture_tab, learning_tab = st.tabs(
        ["Overview", "Architecture", "Learning"]
    )

    with overview:
        st.subheader("Project Idea")
        st.markdown(
            """
This project is a local-first AI teaching assistant for network security
coursework. It turns lecture slides, textbook excerpts, and homework material
into a searchable knowledge base, then uses Retrieval-Augmented Generation to
answer questions and create practice quizzes with visible source citations.

The main idea is simple but important: students should be able to use an AI
tutor without sending private course documents to a hosted model. The app keeps
the document processing, semantic search, and Ollama LLM calls on the local
machine while still delivering an interactive study experience.
"""
        )

        left, middle, right, fourth = st.columns(4)
        left.metric("Knowledge chunks", doc_count)
        middle.metric("Source PDFs", pdf_count)
        right.metric("Processed notes", processed_count)
        fourth.metric("LLM status", "Ready" if ollama_status else "Offline")

        download_left, download_right = st.columns(2)
        for column, (label, path) in zip([download_left, download_right], PROJECT_DOCS.items()):
            with column:
                if path.exists():
                    st.download_button(
                        f"Download {label.lower()}",
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime="text/markdown",
                        use_container_width=True,
                    )

        st.subheader("What the Project Does")
        st.markdown(
            """
- Indexes local PDFs into a ChromaDB vector database.
- Retrieves the most relevant source pages for a question or quiz topic.
- Uses Ollama with `qwen2.5:7b-instruct` when the local model is available.
- Shows citations so answers can be checked against the original material.
- Supports tutor-style explanations and quiz generation in the local codebase.
"""
        )

        if not collection:
            st.warning("The vector database is not initialized yet.")
            st.write("Click **Process PDFs** in the sidebar to index the documents in `Source/`.")

        st.subheader("Source Material Snapshot")
        sample_docs = [p.stem for p in source_files[:8]]
        if sample_docs:
            st.write(", ".join(sample_docs))
            if len(source_files) > len(sample_docs):
                st.caption(f"Showing 8 of {len(source_files)} local PDF sources.")
        else:
            st.info("No source PDFs are currently available in the Source folder.")

    with architecture_tab:
        st.header("How the RAG System Works")
        st.markdown(
            """
The app separates the workflow into four practical layers: document ingestion,
retrieval, generation, and presentation. This keeps the system explainable and
makes it easier to debug each part of the RAG pipeline.
"""
        )

        if DIAGRAM_PATH.exists():
            diagram_text = DIAGRAM_PATH.read_text(encoding="utf-8")
            render_mermaid_diagram(diagram_text)

            with st.expander("View Mermaid source"):
                st.code(diagram_text, language="mermaid")

            st.download_button(
                "Download architecture diagram",
                data=diagram_text,
                file_name=DIAGRAM_PATH.name,
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.warning("Architecture diagram file is missing.")

        flow = [
            ("1. Ingest", "PDF pages are extracted with PyMuPDF and saved as Markdown notes."),
            ("2. Embed", "Page-level chunks are stored in ChromaDB with source and page metadata."),
            ("3. Retrieve", "A user question is matched against the local vector database."),
            ("4. Generate", "Ollama receives the retrieved context and writes the answer or quiz."),
            ("5. Verify", "The interface displays citations so the result can be checked."),
        ]

        for title, body in flow:
            with st.container(border=True):
                st.markdown(f"#### {title}")
                st.write(body)

        st.subheader("Local-First Design")
        st.markdown(
            """
The strongest part of this project is its privacy-preserving design. Course
documents stay in the repository, embeddings are stored in the local ChromaDB
folder, and the LLM endpoint points to `http://localhost:11434`. That makes the
project useful for learning, but also relevant to real AI engineering work
where data boundaries and auditability matter.
"""
        )

    with learning_tab:
        st.header("Learning")
        st.markdown(
            """
This application demonstrates a privacy-preserving Retrieval-Augmented
Generation workflow. Instead of sending course documents to an external model,
the system indexes local PDFs, retrieves the most relevant pages for a question,
and uses a local LLM to compose an answer or quiz.

The important engineering idea is grounding: the model is not expected to answer
from memory alone. It receives retrieved source passages and the interface keeps
those citations visible for verification.

Overall, the project connects several job-relevant AI engineering skills:
document processing, vector search, prompt design, local model integration, and
user-facing evaluation through citations and quiz feedback. It is designed as a
teaching tool, but the same pattern applies to enterprise knowledge assistants,
internal documentation search, and private-domain AI workflows.
"""
        )

        left, right = st.columns(2)
        with left:
            st.markdown(
                """
#### What is being demonstrated

- Local document ingestion from PDFs
- Page-level text extraction with metadata
- Semantic search through ChromaDB
- Local LLM generation through Ollama
- Citation-first tutor and quiz workflows
"""
            )
        with right:
            st.markdown(
                """
#### Why it matters

- Keeps private course material on the user's machine
- Makes answers traceable to source documents
- Supports self-study with immediate quiz feedback
- Shows a practical RAG architecture beyond a static notebook
"""
            )


# Entry point: Run main() when script is executed directly
if __name__ == "__main__":
    main()
