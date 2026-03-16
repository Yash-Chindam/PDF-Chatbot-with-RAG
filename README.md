# PDF Chatbot with RAG (Retrieval-Augmented Generation)

A comprehensive project implementing PDF-based question-answering systems using Retrieval-Augmented Generation (RAG) with multiple LLM backends including FLAN-T5 and Mistral Nemo.

## 📋 Overview

This project provides three different implementations of PDF chatbots, each demonstrating RAG techniques:

1. **PDF Query using RAG with FLAN-T5**: Lightweight, fast retrieval-augmented QA
2. **PDF Query using RAG with Mistral Nemo**: Advanced conversational AI with better reasoning
3. **PDF Chatbot with Mistral Nemo**: Full-featured chatbot with context management

All use FAISS vector database for efficient document retrieval and semantic search.

## 🎯 Objectives

- Build PDF-based question-answering systems
- Implement Retrieval-Augmented Generation (RAG)
- Compare different LLM backends
- Demonstrate vector-based document retrieval
- Create context-aware chatbots
- Enable multi-document querying
- Evaluate answer quality and relevance

## 🗂️ Project Structure

```
PDF-Chatbot-with-RAG/
├── Pdf_chatbot_using_RAG_and_Faiss_with_Mistral_Nemo.ipynb  # Full chatbot
├── Pdf_query_using_RAG_and_Faiss_with_Mistral_Nemo.ipynb    # Query-based QA
├── Pdf_query_using_RAG_and_Faiss_with_FLAN_T5.ipynb         # Lightweight version
├── README.md                                                  # Documentation
└── data/                                                      # PDF files (optional)
```

## 🛠️ Technologies & Libraries

- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Sentence-Transformers
- **LLMs**: 
  - FLAN-T5 Base/Large
  - Mistral Nemo
  - Groq API
  
- **PDF Processing**: 
  - PyPDF2
  - pdf2image
  - LangChain
  
- **Frameworks**: 
  - Transformers
  - PyTorch
  - LangChain

## 📊 Key Features

- **Retrieval-Augmented Generation**:
  - Vector-based document search
  - Context injection into prompts
  - Grounded answer generation
  - Citation capability

- **Multiple Implementations**:
  - Fast lightweight version (FLAN-T5)
  - Advanced conversational version (Mistral)
  - Full chatbot with memory

- **Document Processing**:
  - Extract text from PDFs
  - Split documents into chunks
  - Create embeddings
  - Store in FAISS index

- **Query Capabilities**:
  - Single question answering
  - Multi-turn conversations
  - Context preservation
  - Confidence scoring

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- GPU optional (faster inference)
- PDF files for testing

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd PDF-Chatbot-with-RAG
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open notebooks:
   ```bash
   jupyter notebook
   ```

## 📈 RAG Pipeline

```
PDF Upload
    ↓
Text Extraction
    ↓
Document Chunking
    ↓
Embedding Generation
    ↓
FAISS Index Creation
    ↓
User Query
    ↓
Query Embedding
    ↓
Similarity Search (FAISS)
    ↓
Retrieve Top-K Documents
    ↓
Prompt Construction
    ↓
LLM Generation
    ↓
Answer + Source Citation
```

## 💾 Configuration

### FAISS Setup
```python
from sentence_transformers import SentenceTransformer
import faiss

# Initialize embedder
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Create FAISS index
embeddings = embedder.encode(document_chunks)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
```

### FLAN-T5 Configuration
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

### Mistral Nemo Configuration
```python
from groq import Groq

client = Groq(api_key="your-groq-api-key")
model = "mistral-7b-instruct"
```

## 📝 Usage Examples

### FLAN-T5 RAG Query

```python
# Load PDF and create index
documents = load_pdf("document.pdf")
chunks = split_documents(documents)
embeddings = embedder.encode(chunks)
index = create_faiss_index(embeddings)

# Query
question = "What is the main topic?"
query_embedding = embedder.encode(question)
distances, indices = index.search([query_embedding], k=5)

# Get relevant chunks
relevant_chunks = [chunks[i] for i in indices[0]]
context = " ".join(relevant_chunks)

# Generate answer with FLAN-T5
prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
answer = model.generate(
    **tokenizer(prompt, return_tensors="pt")
)[0]
```

### Mistral Nemo Chatbot

```python
# Initialize Groq client
client = Groq()

# Multi-turn conversation
conversation_history = []

while True:
    user_query = input("You: ")
    
    # Retrieve relevant context
    query_embedding = embedder.encode(user_query)
    _, indices = index.search([query_embedding], k=5)
    context = " ".join([chunks[i] for i in indices[0]])
    
    # Build prompt with history and context
    messages = conversation_history + [
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {user_query}"
        }
    ]
    
    # Generate response
    response = client.messages.create(
        model="mistral-7b-instruct",
        messages=messages,
        max_tokens=512
    )
    
    answer = response.content[0].text
    conversation_history.append({"role": "assistant", "content": answer})
    print(f"Chatbot: {answer}")
```

## ⚙️ Key Parameters

```python
# Document processing
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Retrieval
TOP_K = 5  # Retrieve top 5 chunks
SIMILARITY_THRESHOLD = 0.5

# Generation
MAX_TOKENS = 500
TEMPERATURE = 0.7
```

## 📊 Model Comparison

| Model | Speed | Quality | Memory | Use Case |
|-------|-------|---------|--------|----------|
| FLAN-T5 | ⚡⚡⚡ | Good | Low | Quick answers, lightweight |
| Mistral Nemo | ⚡ | Excellent | Medium | Advanced reasoning, conversations |

## 🔍 Best Practices

- Optimize chunk size for your domain
- Use appropriate embedding model
- Set reasonable TOP_K for context
- Include source citations
- Validate retrieved context quality
- Monitor answer relevance
- Test on diverse queries

## 🎯 Advanced Features

- **Multi-document Indexing**: Query across multiple PDFs
- **Metadata Filtering**: Filter by date, author, category
- **Dynamic Chunking**: Chunk based on document structure
- **Answer Confidence**: Score answers by confidence
- **Source Citation**: Link answers back to PDF pages
- **Conversation Memory**: Maintain context across turns

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## 📚 References

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)
- [FLAN-T5: Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)
- [Mistral 7B](https://mistral.ai/)

## 📄 License

This project is open source and available under the MIT License.

## ⚠️ Important Notes

- API usage will incur costs (Groq/LLM providers)
- Store API keys securely in .env files
- Test retrieval quality on your specific PDFs
- Evaluate answer accuracy for your use case
- Consider document preprocessing needs

## ✉️ Contact

For questions or suggestions, please open an issue or contact the project maintainers.