import torch
import torchaudio
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# python transcribe_audio.py 19-198-0000.flac --output my_output.csv

torchaudio.set_audio_backend("soundfile")

class AudioTranscriber:
    def __init__(self, model_name="facebook/wav2vec2-large-960h", device=None):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device).eval()

    def _load_and_resample_audio(self, audio_path, target_sr=16000):
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.squeeze()

        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)

        return waveform.numpy()

    def transcribe(self, audio_path):
        speech = self._load_and_resample_audio(audio_path)
        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(self.device)

        with torch.inference_mode(), torch.cuda.amp.autocast():
            logits = self.model(inputs).logits

        predicted_ids = torch.argmax(logits[0], dim=-1)
        transcription = self.processor.decode(predicted_ids.cpu())

        print(f"✅ Transcribed: {audio_path}")
        print(f"📝 Output: {transcription}")

        return {
            "audio_path": audio_path,
            "new_text": transcription,
            "wer": None,
            "bleu": None
        }

    def transcribe_to_csv(self, audio_path, output_csv="single_audio_result.csv"):
        result = self.transcribe(audio_path)
        df = pd.DataFrame([result])
        df.to_csv(output_csv, index=False)
        print(f"📁 Saved result to: {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe an audio file using Wav2Vec2.")
    parser.add_argument("audio_path", type=str, help="Path to the audio file (wav format)")
    parser.add_argument("--output", type=str, default="single_audio_result.csv", help="Output CSV file path")

    args = parser.parse_args()

    transcriber = AudioTranscriber()
    transcriber.transcribe_to_csv(args.audio_path, args.output)
