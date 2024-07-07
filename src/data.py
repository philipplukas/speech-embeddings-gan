class HuggingFaceCommonVoiceDataset(Dataset):
    def __init__(self, split, language, speech_embeddings):
        self.dataset = load_dataset("mozilla-foundation/common_voice_11_0", language, split=split)
        self.speech_embeddings = speech_embeddings

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio = self.dataset[idx]['audio']
        audio_path = audio['path']
        embeddings = self.speech_embeddings.get_embeddings_from_speech(audio_path)
        return embeddings
