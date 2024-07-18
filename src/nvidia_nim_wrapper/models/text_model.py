from typing import Union, Generator
from openai import OpenAI
from .base_model import Model

class TextModel(Model):
    def __init__(self, client: OpenAI, model_name: str):
        self.client = client
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> Union[str, Generator[str, None, None]]:
        messages = [{"role": "user", "content": prompt}]
        stream = kwargs.pop('stream', False)
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=stream,
            **kwargs
        )

        if stream:
            return self._stream_response(completion)
        else:
            return completion.choices[0].message.content

    def _stream_response(self, completion) -> Generator[str, None, None]:
        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
