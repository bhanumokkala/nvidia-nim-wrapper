from enum import Enum

class ModelType(Enum):
    TEXT = 'text'
    VISION = 'vision'

from .text_model import TextModel
from .vision_model import VisionModel

__all__ = ['ModelType', 'TextModel', 'VisionModel']
