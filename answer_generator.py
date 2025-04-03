from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import util
import numpy as np

def generate_answer(question, retrieved_info, document_features):
    # Generate high-quality answers using better retrieval and generation techniques
    model_name = "facebook/opt-1.3b"  # More capable than GPT-2, similar local requirements
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 1. IMPROVED RETRIEVAL: Re-rank the results based on semantic similarity to ensure we have the most relevant content
    if len(retrieved_info["text_results"]) > 0:
        # Convert retrieved text to embeddings for better ranking
        question_embedding = document_features["text_features"][0]["chunks"][0]["embedding"]
        
        # Re-rank text results
        for item in retrieved_info["text_results"]:
            # Find the embedding for this text chunk
            for page in document_features["text_features"]:
                if page["page"] == item["page"]:
                    for chunk in page["chunks"]:
                        if chunk["text"] == item["text"]:
                            item["exact_similarity"] = util.pytorch_cos_sim(
                                question_embedding, 
                                chunk["embedding"]
                            ).item()
                            break
        
        # Sort by exact similarity
        retrieved_info["text_results"].sort(key=lambda x: x.get("exact_similarity", 0), reverse=True)
    
    # 2. SMART CONTEXT BUILDING: Use top results but ensure proper context
    max_input_length = 2048  # For larger models like OPT
    reserved_space = 512  # Space for generated answer
    
    usable_input_length = max_input_length - reserved_space
    
    # First, add the highest quality, most relevant text chunks
    context_chunks = []
    token_count = 0
    
    # Add only the most relevant text results (top 3)
    for item in retrieved_info["text_results"][:3]:
        text = f"From page {item['page']+1}: {item['text']}"
        tokens = tokenizer.encode(text)
        if token_count + len(tokens) <= usable_input_length:
            context_chunks.append(text)
            token_count += len(tokens)
    
    # Add the most relevant image captions (top 2)
    for item in retrieved_info["image_results"][:2]:
        caption = f"Image on page {item['page']+1}: {item['caption']}"
        tokens = tokenizer.encode(caption)
        if token_count + len(tokens) <= usable_input_length:
            context_chunks.append(caption)
            token_count += len(tokens)
    
    # 3. BUILD A CLEAR, FOCUSED PROMPT
    context = "\n\n".join(context_chunks)
    
    prompt = f"""Answer the following question using only information from the provided context. 
If you cannot find the answer in the context, simply state that you don't have enough information.
Be concise, accurate, and use only facts from the context.

Context:
{context}

Question: {question}

Answer:"""

    # 4. GENERATE WITH BETTER PARAMETERS
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=300,
            num_return_sequences=1,
            temperature=0.3,  # Lower temperature for more factual responses
            do_sample=True,
            no_repeat_ngram_size=3,  # Prevent repetitive text
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract only the generated answer
    answer_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    # Post-process to remove any repetition and ensure coherence
    answer_text = answer_text.strip()
    if not answer_text:
        answer_text = "Based on the document content, I don't have enough information to answer this question accurately."
    
    # Remove repetitive sentences (a common issue with these models)
    sentences = answer_text.split('. ')
    unique_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence not in unique_sentences:
            unique_sentences.append(sentence)
    
    final_answer = '. '.join(unique_sentences)
    if not final_answer.endswith('.'):
        final_answer += '.'
    
    return {
        "question": question,
        "answer": final_answer,
        "supporting_text": retrieved_info["text_results"][:3],  # Only return the most relevant ones
        "supporting_images": retrieved_info["image_results"][:2]
    }