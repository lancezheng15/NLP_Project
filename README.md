# Speech Analysis and Summarization Pipeline

A comprehensive natural language processing pipeline that combines speech recognition, entity analysis, and text summarization. This project was developed as part of a Text Analytics course, focusing on processing and analyzing audio content using state-of-the-art ML models.

## Project Overview

This project implements an end-to-end pipeline for processing audio content, with three main components:

1. **Speech Recognition**: Uses Facebook's Wav2Vec2 model to convert speech to text
2. **Entity Analysis**: Employs spaCy for named entity recognition and analysis
3. **Text Summarization**: Leverages FLAN-T5 for generating multi-level summaries

The pipeline was initially developed and tested using the LibriSpeech dataset, achieving high accuracy in transcription and meaningful entity extraction and summarization results.

## Project Structure

```
├── app.py                    # Streamlit web interface
├── transcribe_audio.py       # Speech recognition module
├── entity_analyzer.py        # Named entity recognition module
├── text_summarizer.py        # Text summarization module
├── text_restorer.py         # Text preprocessing module
├── process_transcripts.py    # Transcript processing utilities
├── data_processor.py        # Data handling utilities
├── requirements.txt         # Python dependencies
└── environment.yml          # Conda environment specification
```

## Core Components

### 1. Audio Transcription (`transcribe_audio.py`)
- Implements `AudioTranscriber` class using Wav2Vec2
- Supports multiple audio formats
- Automatic device selection (CUDA/MPS/CPU)
- Includes resampling and audio preprocessing
- Outputs results in CSV format

### 2. Entity Analysis (`entity_analyzer.py`)
- `EntityAnalyzer` class powered by spaCy
- Extracts and categorizes named entities
- Provides detailed entity statistics:
  - Entity type distribution
  - Unique entity counts
  - Overall entity statistics

### 3. Text Summarization (`text_summarizer.py`)
- Uses FLAN-T5 for advanced summarization
- Generates multiple summary levels:
  - Short summaries (30-50 words)
  - Long summaries (50-150 words)
- Adaptive length based on input text
- Batch processing support for DataFrames

### 4. Web Interface (`app.py`)
- Interactive Streamlit dashboard
- Real-time audio processing
- Visualization of entity distribution
- Downloadable results in multiple formats
- Support for various audio input formats

## Installation


1. Set up the environment (choose one):

Using conda:
```bash
conda env create -f environment.yml
conda activate speech-analysis
```

Using pip:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### Web Interface
```bash
streamlit run app.py
```

### Command Line Usage

1. Transcribe audio:
```bash
python transcribe_audio.py path/to/audio.wav --output results.csv
```

2. Process entities and generate summaries:
```python
from entity_analyzer import EntityAnalyzer
from text_summarizer import TextSummarizer

# Initialize components
analyzer = EntityAnalyzer()
summarizer = TextSummarizer()

# Process text
entities = analyzer.extract_entities(text)
short_summary, long_summary = summarizer.generate_summaries(text)
```

## System Requirements

- Python 3.10+
- 4GB+ RAM
- GPU Support:
  - CUDA-compatible GPU (optional)
  - Apple M1/M2 chip (MPS support)
- Storage: 
  - ~5GB for models and dependencies
  - Additional space for audio processing

## Key Dependencies

- torch==2.7.0
- transformers==4.30.2
- spacy==3.8.7
- streamlit==1.24.0
- torchaudio (for audio processing)
- pandas (for data handling)
- plotly (for visualizations)

## Performance Notes

- First run downloads required models
- Processing time depends on:
  - Audio file length
  - Selected device (GPU/CPU)
  - Chosen summarization length
- GPU acceleration recommended for batch processing

## Future Improvements

1. Integration with real-time audio streaming
2. Support for additional languages
3. Custom model fine-tuning options
4. Enhanced entity visualization
5. Batch processing optimization


## Contributors

Yunze Wei, Lanfeng Zheng, Keyu Shen, Bo Zhao, Kaiyuan Deng
