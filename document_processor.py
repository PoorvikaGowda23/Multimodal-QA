import fitz  # PyMuPDF
import io
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from utils.image_captioning import generate_image_caption
from utils.text_chunking import chunk_text
import logging

logger = logging.getLogger(__name__)

def preprocess_document(document_path):
    if document_path.endswith('.pdf'):
        return process_pdf(document_path)
    elif document_path.endswith(('.jpg', '.png', '.jpeg')):
        return process_image(document_path)
    else:
        raise ValueError(f"Unsupported document format: {document_path}")

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_content = []
    images = []
    
    try:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            text_content.append({"page": page_num, "text": text})
            
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                images.append({
                    "page": page_num,
                    "index": img_index,
                    "image": image,
                    "location": {"x": img[1], "y": img[2], "width": img[3], "height": img[4]}
                })
    finally:
        doc.close()
    return {"text": text_content, "images": images}

def process_image(image_path):
    try:
        image = Image.open(image_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        text = pytesseract.image_to_string(image)
        return {
            "text": [{"page": 0, "text": text}],
            "images": [{
                "page": 0,
                "index": 0,
                "image": image,
                "location": {"x": 0, "y": 0, "width": image.width, "height": image.height}
            }]
        }
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")

def extract_image_features(images):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_features = []
    for img_data in images:
        try:
            img = img_data["image"]
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                features = feature_extractor(img_tensor)
                features = features.flatten()
            caption = generate_image_caption(img)
            image_features.append({
                "page": img_data["page"],
                "index": img_data["index"],
                "features": features,
                "location": img_data["location"],
                "caption": caption
            })
        except Exception as e:
            logger.warning(f"Skipping image on page {img_data['page']}: {str(e)}")
            continue
    return image_features

def extract_text_features(text_content):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    features = []
    for page in text_content:
        chunks = chunk_text(page["text"])
        chunk_embeddings = []
        for chunk in chunks:
            embedding = model.encode(chunk, convert_to_tensor=True)
            chunk_embeddings.append({
                "text": chunk,
                "embedding": embedding
            })
        features.append({
            "page": page["page"],
            "chunks": chunk_embeddings
        })
    return features

def extract_features(preprocessed_doc):
    text_features = extract_text_features(preprocessed_doc["text"])
    image_features = extract_image_features(preprocessed_doc["images"])
    model = SentenceTransformer('all-MiniLM-L6-v2')
    for img in image_features:
        if "caption" in img:
            img["caption_embedding"] = model.encode(img["caption"], convert_to_tensor=True)
    return {
        "text_features": text_features,
        "image_features": image_features,
        "raw_document": preprocessed_doc
    }