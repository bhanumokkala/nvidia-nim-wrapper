from typing import Dict, Union, Generator
from .models import ModelType, TextModel, VisionModel
from .utils.config import load_config
from openai import OpenAI

class NvidiaNIMWrapper:
    def __init__(self, config_path: str = 'config.yaml', openai_client=None):
        self.config = load_config(config_path)
        self.api_key = self.config['api_key']
        self.text_base_url = self.config['text_base_url']
        self.vision_base_url = self.config['vision_base_url']
        self.openai_client = openai_client or OpenAI(base_url=self.text_base_url, api_key=self.api_key)
        self.models: Dict[str, Union[TextModel, VisionModel]] = {}

    def register_model(self, model_name: str, model_type: ModelType):
        if model_type == ModelType.TEXT:
            self.models[model_name] = TextModel(self.openai_client, model_name)
        elif model_type == ModelType.VISION:
            self.models[model_name] = VisionModel(self.api_key, self.vision_base_url, model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def generate(self, model_name: str, prompt: str, **kwargs) -> Union[str, Dict, Generator[str, None, None]]:
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered. Please register the model first.")
        
        return self.models[model_name].generate(prompt, **kwargs)

    def generate_text(self, model_name: str, prompt: str, **kwargs) -> str:
        response = self.generate(model_name, prompt, **kwargs)
        if isinstance(response, Generator):
            return ''.join(response)
        return response

    def generate_vision(self, model_name: str, prompt: str, image_path: str, **kwargs) -> Dict:
        return self.generate(model_name, prompt, image_path=image_path, **kwargs)