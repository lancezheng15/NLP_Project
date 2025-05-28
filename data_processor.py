import os
import pandas as pd

def create_dataset_from_audio_dir(base_dir, output_csv="full_transcripts.csv"):
    """
    Create a dataset from a directory of audio files and their transcripts.
    
    Args:
        base_dir (str): Root path to the unzipped dataset
        output_csv (str): Path to save the output CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing the dataset
    """
    data = []
    
    # Loop through speaker folders
    for speaker_id in os.listdir(base_dir):
        speaker_path = os.path.join(base_dir, speaker_id)
        if not os.path.isdir(speaker_path):
            continue

        # Loop through chapter folders
        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)
            if not os.path.isdir(chapter_path):
                continue

            # Transcript file
            trans_path = os.path.join(chapter_path, f"{speaker_id}-{chapter_id}.trans.txt")
            if not os.path.exists(trans_path):
                continue

            # Read each line in transcript and match with audio
            with open(trans_path, "r") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) != 2:
                        continue
                    utt_id, text = parts
                    flac_path = os.path.join(chapter_path, utt_id + ".flac")
                    if os.path.exists(flac_path):
                        data.append({
                            "speaker_id": speaker_id,
                            "chapter_id": chapter_id,
                            "utterance_id": utt_id,
                            "text": text,
                            "audio_path": flac_path
                        })

    # Create full DataFrame
    df = pd.DataFrame(data)
    if output_csv:
        df.to_csv(output_csv, index=False)
    return df

def load_transcripts(csv_path):
    """
    Load transcripts from a CSV file and perform basic preprocessing.
    
    Args:
        csv_path (str): Path to the CSV file containing transcripts
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    df = pd.read_csv(csv_path)
    
    # Remove any rows with missing transcriptions
    if 'new_text' in df.columns:
        df = df[df['new_text'].notna()]
    
    return df

def save_transcripts(df, output_path):
    """
    Save transcripts to a CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame containing transcripts
        output_path (str): Path to save the CSV file
    """
    df.to_csv(output_path, index=False) 