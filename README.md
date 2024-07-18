# NVIDIA NIM Wrapper

A unified API wrapper for NVIDIA Inference Microservices (NIM).

## Installation

```bash
pip install nvidia-nim-wrapper
```

## Usage

```python
from nvidia_nim_wrapper import NvidiaNIMWrapper, ModelType

wrapper = NvidiaNIMWrapper('config.yaml')

# Register models
wrapper.register_model("microsoft/phi-3-medium-128k-instruct", ModelType.TEXT)
wrapper.register_model("microsoft/phi-3-vision-128k-instruct", ModelType.VISION)

# Generate text
text_response = wrapper.generate_text(
    "microsoft/phi-3-medium-128k-instruct",
    "What is the capital of France?",
    temperature=0.7,
    max_tokens=100
)
print(text_response)

# Generate vision
vision_response = wrapper.generate_vision(
    "microsoft/phi-3-vision-128k-instruct",
    "Describe this image:",
    image_path="path/to/image.jpg",
    max_tokens=200
)
print(vision_response)
```

## Configuration

Create a `config.yaml` file in your project root with the following structure:

```yaml
api_key: your_api_key_here
text_base_url: https://integrate.api.nvidia.com/v1
vision_base_url: https://ai.api.nvidia.com/v1/vlm
```

## Development

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix or MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run tests: `python -m unittest discover tests`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
