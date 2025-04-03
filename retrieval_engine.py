import torch
import torch.nn.functional as F

def retrieve_relevant_information(document_features, processed_question):

    # Retrieve relevant text chunks and images based on the question
    text_results = retrieve_text(document_features["text_features"], processed_question)
    
    image_results = []
    if processed_question["requires_image"]:
        image_results = retrieve_images(document_features["image_features"], processed_question)
    
    return {
        "text_results": text_results,
        "image_results": image_results
    }

def retrieve_text(text_features, processed_question):

    # Retrieve relevant text chunks using semantic search
    results = []
    
    # Use the projected 384-dim embedding for text retrieval
    question_embedding = processed_question["embedding"]
    
    for page in text_features:
        for chunk in page["chunks"]:
            # Calculate cosine similarity
            similarity = F.cosine_similarity(
                question_embedding,
                chunk["embedding"].unsqueeze(0)
            ).item()
            
            results.append({
                "page": page["page"],
                "text": chunk["text"],
                "similarity": similarity
            })
    
    # Sort by similarity and take top results
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:5]

def retrieve_images(image_features, processed_question):
    # Retrieve relevant images based on the question
    results = []
    
    # Use the direct SentenceTransformer embedding for image caption comparison
    question_text_embedding = processed_question["question_text_embedding"]
    
    for img in image_features:
        if "caption" in img and img["caption"]:
            # Calculate similarity with image caption
            caption_embedding = img.get("caption_embedding")  # Should be precomputed
            if caption_embedding is None:
                continue
                
            similarity = F.cosine_similarity(
                question_text_embedding.unsqueeze(0),
                caption_embedding.unsqueeze(0)
            ).item()
            
            results.append({
                "page": img["page"],
                "index": img["index"],
                "similarity": similarity,
                "caption": img["caption"]
            })
    
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:3]