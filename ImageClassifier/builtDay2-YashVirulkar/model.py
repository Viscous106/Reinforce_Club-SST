from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def get_model_and_processor():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def get_detailed_model_and_processor():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

def describe_image(processor, model, image_path):

    try:
        # Open the image
        raw_image = Image.open(image_path).convert("RGB")

        # Preprocess the image
        inputs = processor(raw_image, return_tensors="pt")

        # Generate a caption
        out = model.generate(**inputs)

        # Decode the caption
        caption = processor.decode(out[0], skip_special_tokens=True)

        return caption
    except Exception as e:
        return f"An error occurred: {e}"

def describe_image_in_detail(processor, model, image_path):
    try:
        # Open the image
        raw_image = Image.open(image_path).convert("RGB")

        # Preprocess the image
        inputs = processor(raw_image, return_tensors="pt")

        # Generate a caption
        out = model.generate(**inputs, max_length=100, num_beams=5, early_stopping=True)

        # Decode the caption
        caption = processor.decode(out[0], skip_special_tokens=True)

        return caption
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    dummy_image = Image.new('RGB', (60, 30), color = 'red')
    dummy_image.save("dummy_image.png")

    processor, model = get_model_and_processor()
    description = describe_image(processor, model, "dummy_image.png")
    print(f"Generated description: {description}")

    detailed_processor, detailed_model = get_detailed_model_and_processor()
    detailed_description = describe_image_in_detail(detailed_processor, detailed_model, "dummy_image.png")
    print(f"Generated detailed description: {detailed_description}")