import unittest
from src.embeddings import SpeechEmbeddings

class TestSpeechEmbeddings(unittest.TestCase):
    def test_get_embeddings_from_speech(self):
        embeddings = SpeechEmbeddings()
        result = embeddings.get_embeddings_from_speech("path_to_audio_file.wav")
        self.assertIsNotNone(result)

    def test_turn_embeddings_into_speech(self):
        embeddings = SpeechEmbeddings()
        embeddings_data = torch.randn((1, 512))  # Mock embeddings
        embeddings.turn_embeddings_into_speech(embeddings_data, "output_audio_file.wav")

if __name__ == "__main__":
    unittest.main()

