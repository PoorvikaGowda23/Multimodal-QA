''' 7. Main System Integration '''
import os
import json
import logging
from flask import Flask, request, jsonify, render_template
import tempfile

# Import modules
from document_processor import preprocess_document, extract_features
from question_processor import process_question
from retrieval_engine import retrieve_relevant_information
from answer_generator import generate_answer
from audio_processor import transcribe_audio, generate_speech_response, save_temp_audio_file


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Store processed documents in memory (in production, use a database)
document_cache = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        logger.error("No file selected")
        return jsonify({"error": "No selected file"}), 400
        
    # Validate file extension
    allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        logger.error(f"Unsupported file type: {file_ext}")
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"}), 400

    temp_file = None
    try:
        # Create a secure temporary file
        temp_file = tempfile.NamedTemporaryFile(
            suffix=file_ext, 
            delete=False,
            dir='temp'  # Store in temp directory
        )
        os.makedirs('temp', exist_ok=True)
        
        # Save the uploaded file to temp location
        file.save(temp_file.name)
        logger.info(f"Temporarily saved file to {temp_file.name}")

        # Process the document
        logger.info("Processing document...")
        preprocessed_doc = preprocess_document(temp_file.name)
        features = extract_features(preprocessed_doc)
        
        # Generate a document ID and store in cache
        doc_id = str(abs(hash(file.filename + str(os.path.getmtime(temp_file.name)))))
        document_cache[doc_id] = features
        logger.info(f"Document processed successfully. ID: {doc_id}")

        return jsonify({
            "status": "success",
            "document_id": doc_id,
            "pages": len(preprocessed_doc["text"]),
            "images": len(preprocessed_doc["images"]),
            "filename": file.filename
        })
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to process document: {str(e)}"}), 500
        
    finally:
        # Clean up temporary file
        if temp_file:
            try:
                temp_file.close()  # Close the file handle first
                os.unlink(temp_file.name)
            except Exception as e:
                logger.warning(f"Could not delete temp file {temp_file.name}: {str(e)}")
       

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    if not data or 'question' not in data or 'document_id' not in data:
        logger.error("Missing question or document_id in request")
        return jsonify({"error": "Missing question or document_id"}), 400
        
    question = data['question']
    doc_id = data['document_id']
    
    if doc_id not in document_cache:
        logger.error(f"Document ID not found: {doc_id}")
        return jsonify({"error": "Document not found, please upload it first"}), 404
        
    try:
        logger.info(f"Processing question: {question}")
        
        # Process the question
        processed_question = process_question(question)
        
        # Retrieve relevant information
        retrieved_info = retrieve_relevant_information(
            document_cache[doc_id], 
            processed_question
        )
        
        # Generate answer
        result = generate_answer(
            question, 
            retrieved_info,
            document_cache[doc_id]
        )
        
        logger.info("Successfully generated answer")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    

@app.route('/ask_voice', methods=['POST'])
def ask_voice():
    if 'audio' not in request.files or 'document_id' not in request.form:
        logger.error("Missing audio file or document_id in request")
        return jsonify({"error": "Missing audio file or document_id"}), 400

    audio_file = request.files['audio']
    doc_id = request.form['document_id']

    if doc_id not in document_cache:
        logger.error(f"Document ID not found: {doc_id}")
        return jsonify({"error": "Document not found, please upload it first"}), 404

    try:
        # Save audio file to a temporary location
        temp_audio_path = save_temp_audio_file(audio_file)
        
        # Transcribe the audio to text
        question_text = transcribe_audio(temp_audio_path)
        
        # Process the question and generate answer (as in your existing logic)
        processed_question = process_question(question_text)
        retrieved_info = retrieve_relevant_information(
            document_cache[doc_id], 
            processed_question
        )
        result = generate_answer(
            question_text, 
            retrieved_info,
            document_cache[doc_id]
        )
        
        answer_text = result.get("answer", "Sorry, I could not generate an answer.")
        
        # Generate audio response from the answer text
        audio_base64 = generate_speech_response(answer_text)
        
        response_payload = {
            "question": question_text,
            "answer": answer_text,
            "audio": audio_base64
        }
        logger.info("Successfully generated voice answer")
        return jsonify(response_payload)
        
    except Exception as e:
        logger.error(f"Error processing voice question: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
        
    finally:
        try:
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
        except Exception as e:
            logger.warning(f"Could not delete temp audio file {temp_audio_path}: {str(e)}")


if __name__ == '__main__':
    # Create temp directory if it doesn't exist
    os.makedirs('temp', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0')