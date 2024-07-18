import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.nvidia_nim_wrapper import NvidiaNIMWrapper, ModelType

import unittest
from unittest.mock import patch, MagicMock

class TestNvidiaNIMWrapper(unittest.TestCase):
    def setUp(self):
        self.config = {
            'api_key': 'test_api_key',
            'text_base_url': 'https://test.text.url',
            'vision_base_url': 'https://test.vision.url'
        }
        self.mock_openai = MagicMock()
        with patch('src.nvidia_nim_wrapper.wrapper.load_config', return_value=self.config):
            self.wrapper = NvidiaNIMWrapper('dummy_config.yaml', openai_client=self.mock_openai)

    def test_register_text_model(self):
        self.wrapper.register_model("test_text_model", ModelType.TEXT)
        self.assertIn("test_text_model", self.wrapper.models)

    def test_register_vision_model(self):
        self.wrapper.register_model("test_vision_model", ModelType.VISION)
        self.assertIn("test_vision_model", self.wrapper.models)

    def test_generate_text(self):
        # Set up the mock OpenAI client
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "Test response"
        self.mock_openai.chat.completions.create.return_value = mock_completion

        self.wrapper.register_model("test_text_model", ModelType.TEXT)
        response = self.wrapper.generate_text("test_text_model", "Test prompt")
        self.assertEqual(response, "Test response")

    @patch('src.nvidia_nim_wrapper.models.vision_model.requests.post')
    @patch('src.nvidia_nim_wrapper.models.vision_model.open', create=True)
    def test_generate_vision(self, mock_open, mock_post):
        mock_open.return_value.__enter__.return_value.read.return_value = b'dummy image content'
        
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Test vision response"}
        mock_post.return_value = mock_response

        self.wrapper.register_model("test_vision_model", ModelType.VISION)
        response = self.wrapper.generate_vision("test_vision_model", "Test prompt", "dummy_image.jpg")
        self.assertEqual(response, {"response": "Test vision response"})

if __name__ == '__main__':
    unittest.main()