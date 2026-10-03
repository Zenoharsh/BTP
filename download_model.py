import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

def download_and_cache_model():
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    
    print(f"Downloading and caching Processor for {model_id}...")
    # Explicitly use AutoProcessor as required by true Vision-Language Models like Qwen2-VL
    processor = AutoProcessor.from_pretrained(model_id)
    print("Processor (including Image Processor and Text Tokenizer) successfully cached.")

    print(f"\nDownloading and caching Model for {model_id} in bfloat16 precision...")
    # Load model in bfloat16 precision to save memory
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("\nModel successfully cached.")
    
    # Verification: Confirm both the vision encoder (ViT) and the text backbone exist
    print("\nVerifying cached components:")
    if hasattr(model, 'visual'):
        print("✅ Success: Vision Encoder (ViT) architecture successfully detected and cached.")
    else:
        print("❌ Error: Vision Encoder not detected. This might not be a true VLM.")
        
    if hasattr(model, 'model'):
        print("✅ Success: Text Backbone architecture successfully detected and cached.")
    else:
        print("❌ Error: Text Backbone not detected.")
        
if __name__ == "__main__":
    download_and_cache_model()
