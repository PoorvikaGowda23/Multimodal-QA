from transformers import BertTokenizer, BertModel
import torch
from sentence_transformers import SentenceTransformer
import torch.nn as nn

class DimensionAdapter(nn.Module):
    # Adapter to convert 768-dim BERT embeddings to 384-dim
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(768, 384)
        
    def forward(self, x):
        return self.projection(x)

bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dimension_adapter = DimensionAdapter()
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def process_question(question_text):

    # Analyze the question using BERT but adapt dimensions to match SentenceTransformer
    # Tokenize and get BERT embeddings
    inputs = tokenizer(question_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    # Get token representation (768-dim)
    bert_embedding = outputs.last_hidden_state[:, 0, :]
    
    # Project to 384-dim to match SentenceTransformer
    question_embedding = dimension_adapter(bert_embedding)
    
    # Classify if question is about text, images, or both
    image_keywords = ['image', 'picture', 'photo', 'figure', 'graph', 'chart', 'diagram', 'visual']
    requires_image = any(keyword in question_text.lower() for keyword in image_keywords)
    
    # For image caption comparison, use SentenceTransformer directly
    question_text_embedding = sentence_model.encode(question_text, convert_to_tensor=True)
    
    return {
        "text": question_text,
        "bert_embedding": bert_embedding,  # Original 768-dim
        "embedding": question_embedding,    # Projected 384-dim
        "question_text_embedding": question_text_embedding,  # For image comparison
        "requires_image": requires_image,
        "requires_text": True
    }