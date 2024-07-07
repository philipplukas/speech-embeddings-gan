import torch
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText, SpeechT5ForTextToSpeech

class SpeechEmbeddings:
    def __init__(self):
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")
        self.model_stt = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr")
        self.model_tts = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        
    def get_embeddings_from_speech(self, audio_path):
        # Load audio file
        speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
        inputs = self.processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt")
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.model_stt(**inputs).last_hidden_state
        
        return embeddings

    def turn_embeddings_into_speech(self, embeddings, output_path):
        # Convert embeddings to speech
        with torch.no_grad():
            generated_speech = self.model_tts.generate(inputs_embeds=embeddings)
        
        # Save generated speech to file
        generated_speech = generated_speech.cpu().numpy()
        librosa.output.write_wav(output_path, generated_speech, sr=16000)

