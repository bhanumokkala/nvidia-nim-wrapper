import os
from src.nvidia_nim_wrapper import NvidiaNIMWrapper, ModelType

def main():
    # Initialize the wrapper with the existing config file
    wrapper = NvidiaNIMWrapper('config.yaml')

    # Register models
    wrapper.register_model("microsoft/phi-3-medium-128k-instruct", ModelType.TEXT)
    wrapper.register_model("meta/llama3-70b-instruct", ModelType.TEXT)
    wrapper.register_model("microsoft/phi-3-vision-128k-instruct", ModelType.VISION)

    # Test text generation with Phi model
    print("Testing Phi model:")
    phi_response = wrapper.generate_text(
        "microsoft/phi-3-medium-128k-instruct",
        "Explain the concept of quantum computing in simple terms.",
        temperature=0.7,
        max_tokens=150
    )
    print(f"Phi model response: {phi_response}\n")

    # Test text generation with Llama model
    print("Testing Llama model:")
    llama_response = wrapper.generate_text(
        "meta/llama3-70b-instruct",
        "What are the main differences between machine learning and deep learning?",
        temperature=0.5,
        max_tokens=200
    )
    print(f"Llama model response: {llama_response}\n")

    # Test vision model
    print("Testing Vision model:")
    # Make sure to replace 'path/to/your/image.jpg' with an actual image path
    vision_response = wrapper.generate_vision(
        "microsoft/phi-3-vision-128k-instruct",
        "Describe the main elements in this image.",
        image_path="merlion.png",
        max_tokens=100
    )
    print(f"Vision model response: {vision_response}")

if __name__ == "__main__":
    main()