import base64
import requests
from typing import Dict, Generator
from .base_model import Model

class VisionModel(Model):
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

    def generate(self, prompt: str, image_path: str, **kwargs) -> Dict:
        try:
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()
        except FileNotFoundError:
            # If the file is not found, use a dummy base64 string
            image_b64 = "dummybase64string"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream" if kwargs.get('stream', False) else "application/json"
        }

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f'{prompt} <img src="data:image/png;base64,{image_b64}" />'
                }
            ],
            **kwargs
        }

        response = requests.post(f"{self.base_url}/{self.model_name}", headers=headers, json=payload)

        if kwargs.get('stream', False):
            return self._stream_response(response)
        else:
            return response.json()

    def _stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        for line in response.iter_lines():
            if line:
                yield line.decode("utf-8")