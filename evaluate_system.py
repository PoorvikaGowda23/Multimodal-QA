from document_processor import preprocess_document, extract_features
from question_processor import process_question
from retrieval_engine import retrieve_relevant_information
from answer_generator import generate_answer


def evaluate_system(test_document_path, test_questions):
    """Evaluate system performance on test questions"""
    # Process document
    preprocessed_doc = preprocess_document(test_document_path)
    features = extract_features(preprocessed_doc)
    
    results = []
    for question in test_questions:
        # Process question
        processed_question = process_question(question["text"])
        
        # Retrieve information
        retrieved_info = retrieve_relevant_information(features, processed_question)
        
        # Generate answer
        result = generate_answer(question["text"], retrieved_info, features)
        
        # Compare with ground truth (if available)
        accuracy = "N/A"
        if "answer" in question:
            # Simple string matching for now
            if question["answer"].lower() in result["answer"].lower():
                accuracy = "Correct"
            else:
                accuracy = "Incorrect"
        
        results.append({
            "question": question["text"],
            "generated_answer": result["answer"],
            "ground_truth": question.get("answer", "N/A"),
            "accuracy": accuracy
        })
    
    return results
