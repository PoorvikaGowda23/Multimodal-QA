# Multimodal Document QA System 📑✨

A Flask-based web application that processes documents (PDFs and images) and answers questions using text and image content. Supports text-based and voice-based queries with real-time responses, leveraging advanced NLP and computer vision techniques.

## 🌟 Features

- **📤 Document Upload**
  - Upload PDFs, PNGs, JPGs, or JPEGs
  - Extracts text and images for processing
- **❓ Question Answering**
  - Ask questions about document content (text and images)
  - Get concise, context-based answers
- **🎤 Voice Interaction**
  - Speak your question and receive an audio response
  - Transcribes audio input and generates speech output
- **🔍 Multimodal Retrieval**
  - Retrieves relevant text chunks and image captions
  - Uses semantic similarity for accurate results
- **💾 In-Memory Storage**
  - Temporarily caches processed documents
  - Secure temporary file handling

## 🛠 Tech Stack

| Component               | Technology                         |
|-------------------------|------------------------------------|
| Frontend                | HTML, Bootstrap, JavaScript        |
| Backend                 | Python 3.9+, Flask                 |
| NLP                     | Transformers (BERT, OPT-1.3B)      |
| Computer Vision         | PyMuPDF, ResNet50, ViT-GPT2        |
| Speech Processing       | SpeechRecognition, gTTS            |
| Embeddings              | SentenceTransformers (MiniLM)      |
| Package Manager         | pip                                |

## 🚀 Getting Started

### Prerequisites
- **Python 3.9+**
- **pip package manager**
- **Tesseract-OCR** (for image text extraction, install separately)

### Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/your-username/multimodal-document-qa.git
    cd multimodal-document-qa
