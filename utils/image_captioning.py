from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

def generate_image_caption(image):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            output_ids = model.generate(pixel_values, max_length=20)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return "No caption available"