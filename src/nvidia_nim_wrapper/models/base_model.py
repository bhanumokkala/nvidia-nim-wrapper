from abc import ABC, abstractmethod
from typing import Union, Dict, Generator

class Model(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Union[str, Dict, Generator[str, None, None]]:
        pass
