import streamlit as st
import pandas as pd
import tempfile
import os
from transcribe_audio import AudioTranscriber
from text_restorer import TextRestorer
from entity_analyzer import EntityAnalyzer
from text_summarizer import TextSummarizer
import plotly.express as px
from collections import Counter
import spacy
import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import numpy as np
from pydub import AudioSegment
from jiwer import wer
from nltk.translate.bleu_score import sentence_bleu
import re

print("Starting app.py execution...")

def normalize_case(text):
    """
    Converts text that appears to be all uppercase to proper case.
    
    Args:
        text (str): Input text, potentially in all uppercase
        
    Returns:
        str: Text converted to proper case if it was primarily uppercase
    """
    if not isinstance(text, str):
        return text
        
    # Check if the text is primarily uppercase (allowing for some punctuation and digits)
    uppercase_ratio = sum(1 for c in text if c.isalpha() and c.isupper()) / max(1, sum(1 for c in text if c.isalpha()))
    
    if uppercase_ratio > 0.8:  # If more than 80% of alphabetic chars are uppercase
        # Convert to lowercase first
        text = text.lower()
        
        # Capitalize first letter of each sentence
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.capitalize() for s in sentences]
        text = ' '.join(sentences)
        
        # Capitalize common proper nouns and I
        text = re.sub(r'\bi\b', 'I', text)
        
    return text

def convert_to_wav(audio_file):
    """Convert uploaded audio to WAV format"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
        # First save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_orig:
            tmp_orig.write(audio_file.getvalue())
            tmp_orig_path = tmp_orig.name

        # Convert to WAV using pydub
        audio = AudioSegment.from_file(tmp_orig_path)
        audio.export(tmp_wav.name, format='wav')
        
        # Clean up original temp file
        os.unlink(tmp_orig_path)
        
        return tmp_wav.name

def load_audio(file_path, target_sr=16000):
    """Load audio file using soundfile and convert to torch tensor"""
    # Read audio file
    audio, sr = sf.read(file_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Convert to float32
    audio = audio.astype(np.float32)
    
    # Resample if necessary
    if sr != target_sr:
        audio_tensor = torch.from_numpy(audio)
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio_tensor = resampler(audio_tensor)
        audio = audio_tensor.numpy()
    
    return audio, target_sr

# Initialize models
@st.cache_resource
def load_models():
    try:
        print("Starting to load models...")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
        
        # Load Wav2Vec2 model (using base model instead of large)
        print("Loading Wav2Vec2 processor...")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", local_files_only=False)
        
        print("Loading Wav2Vec2 model...")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", local_files_only=False)
        
        # Set device for M1 Mac support
        print("Setting up device...")
        if torch.backends.mps.is_available():
            device = "mps"
            print("Using MPS device")
        elif torch.cuda.is_available():
            device = "cuda"
            print("Using CUDA device")
        else:
            device = "cpu"
            print("Using CPU device")
            
        print(f"Moving model to device: {device}")
        model = model.to(device)
        model.eval()
        print("Model loaded and moved to device successfully")
        
    except Exception as e:
        print(f"Error in model loading: {str(e)}")
        st.error(f"Error loading Wav2Vec2 model: {str(e)}")
        st.stop()
    
    # Load spaCy model
    try:
        print("Loading spaCy model...")
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    # Load summarization model
    try:
        print("Loading summarization model...")
        from transformers import pipeline
        # Set device for summarizer
        if torch.backends.mps.is_available():
            device = "mps"
            print("Using MPS device for summarizer")
        elif torch.cuda.is_available():
            device = "cuda"
            print("Using CUDA device for summarizer")
        else:
            device = "cpu"
            print("Using CPU device for summarizer")
            
        print(f"Creating summarization pipeline with device: {device}")
        summarizer = pipeline("summarization", model="google/flan-t5-base", device=device)
        print("Summarization model loaded successfully")
        
    except Exception as e:
        print(f"Error loading summarization model: {str(e)}")
        st.error(f"Error loading summarization model: {str(e)}")
        st.stop()
    
    return processor, model, nlp, summarizer, device

def process_audio(audio_file, processor, model, device):
    """Transcribe audio using Wav2Vec2"""
    try:
        print("Starting audio processing...")
        # Convert to WAV if needed
        wav_path = convert_to_wav(audio_file)
        print(f"Converted to WAV: {wav_path}")
        
        # Load the audio file using soundfile
        print("Loading audio file...")
        speech_array, sampling_rate = load_audio(wav_path)
        print(f"Audio loaded with sampling rate: {sampling_rate}")

        # Process through Wav2Vec2
        print("Processing through Wav2Vec2...")
        inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)
        print(f"Moving inputs to device: {device}")
        inputs = inputs.input_values.to(device)

        # Run inference
        print("Running inference...")
        with torch.inference_mode():
            logits = model(inputs).logits

        # Decode the transcription
        print("Decoding transcription...")
        predicted_ids = torch.argmax(logits[0], dim=-1)
        transcription = processor.decode(predicted_ids.cpu())
        print("Transcription completed successfully")

        # Clean up temporary file
        os.unlink(wav_path)

        return transcription
    except Exception as e:
        print(f"Error in process_audio: {str(e)}")
        st.error(f"Error processing audio: {str(e)}")
        return ""

def main():
    st.set_page_config(
        page_title="Audio Analysis Tool",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Audio Analysis Tool 🎧")
    st.write("Upload an audio file to get transcription, named entities, and summaries!")
    
    # File upload
    audio_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'flac'])
    
    if audio_file is not None:
        # Display audio player
        st.audio(audio_file)
        
        # Process button
        if st.button("Process Audio"):
            with st.spinner("Processing audio..."):
                processor, model, nlp, summarizer, device = load_models()
                transcription = process_audio(audio_file, processor, model, device)
                
                if transcription:
                    # Normalize case
                    normalized_transcription = normalize_case(transcription)
                    
                    # Create tabs for different analyses
                    tab1, tab2, tab3 = st.tabs(["Transcription", "Named Entities", "Summaries"])
                    
                    with tab1:
                        st.subheader("Full Transcription")
                        
                        # Original transcription with styled container
                        st.markdown("#### 📝 Original Transcription")
                        with st.container():
                            st.markdown("""
                            <div style="
                                border-left: 3px solid #FF4B4B;
                                padding-left: 20px;
                                margin: 10px 0;
                                background-color: #f0f2f6;
                                border-radius: 5px;
                                padding: 20px;
                            ">
                            {}
                            </div>
                            """.format(transcription), unsafe_allow_html=True)
                        
                        # Normalized transcription with styled container
                        st.markdown("#### ✨ Normalized Transcription")
                        with st.container():
                            st.markdown("""
                            <div style="
                                border-left: 3px solid #00CC66;
                                padding-left: 20px;
                                margin: 10px 0;
                                background-color: #f0f2f6;
                                border-radius: 5px;
                                padding: 20px;
                            ">
                            {}
                            </div>
                            """.format(normalized_transcription), unsafe_allow_html=True)
                        
                        # Add some space before the download buttons
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Download buttons with improved styling
                        st.markdown("#### 💾 Download Options")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "📥 Download Original",
                                transcription,
                                "original_transcription.txt",
                                "text/plain",
                                use_container_width=True
                            )
                        with col2:
                            st.download_button(
                                "📥 Download Normalized",
                                normalized_transcription,
                                "normalized_transcription.txt",
                                "text/plain",
                                use_container_width=True
                            )
                    
                    with tab2:
                        st.subheader("Named Entities")
                        # Process named entities using spaCy on normalized text
                        doc = nlp(normalized_transcription)
                        entities = [(ent.text, ent.label_) for ent in doc.ents]
                        
                        if entities:
                            # Convert entities to DataFrame for display
                            entities_df = pd.DataFrame(entities, columns=['Text', 'Type'])
                            
                            # Create entity count visualization
                            entity_counts = Counter(entities_df['Type'])
                            fig = px.bar(
                                x=list(entity_counts.keys()),
                                y=list(entity_counts.values()),
                                title="Named Entity Distribution",
                                labels={'x': 'Entity Type', 'y': 'Count'}
                            )
                            st.plotly_chart(fig)
                            
                            # Display entities table
                            st.write("Detailed Entities:")
                            st.dataframe(entities_df)
                            
                            # Download option
                            csv = entities_df.to_csv(index=False)
                            st.download_button(
                                "Download Entities CSV",
                                csv,
                                "entities.csv",
                                "text/csv"
                            )
                        else:
                            st.write("No named entities found in the text.")
                    
                    with tab3:
                        st.subheader("Text Summaries")
                        
                        # Generate summary using the summarizer pipeline on normalized text
                        input_length = len(normalized_transcription.split())
                        max_length = min(200, max(30, input_length // 2))
                        min_length = max(10, input_length // 4)
                        
                        summary = summarizer(
                            normalized_transcription,
                            max_length=max_length,
                            min_length=min_length,
                            do_sample=False
                        )[0]['summary_text']
                        
                        st.write("Summary:")
                        st.write(summary)
                        
                        # Create download button for summary
                        st.download_button(
                            "Download Summary",
                            summary,
                            "summary.txt",
                            "text/plain"
                        )

if __name__ == "__main__":
    main() 