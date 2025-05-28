import argparse
from data_processor import load_transcripts, save_transcripts
from text_restorer import TextRestorer
from entity_analyzer import EntityAnalyzer
from text_summarizer import TextSummarizer

def main(input_csv, output_csv):
    """
    Process transcripts through the pipeline:
    1. Load transcribed data
    2. Restore text case and punctuation
    3. Extract named entities
    4. Generate text summaries
    
    Args:
        input_csv (str): Path to input CSV file with transcripts
        output_csv (str): Path to save the processed output
    """
    print("Loading transcripts...")
    df = load_transcripts(input_csv)
    
    print("Restoring text case and punctuation...")
    restorer = TextRestorer()
    df = restorer.restore_df_text_column(df)
    
    print("Extracting named entities...")
    analyzer = EntityAnalyzer()
    df = analyzer.analyze_df_text_column(df)
    
    # Get entity statistics
    stats = analyzer.get_entity_statistics(df)
    print("\nEntity Statistics:")
    print(f"Total entities found: {stats['total_entities']}")
    print("\nEntity counts by type:")
    for ent_type, count in stats['entity_type_counts'].items():
        print(f"{ent_type}: {count}")
    
    print("\nGenerating text summaries...")
    summarizer = TextSummarizer()
    df = summarizer.summarize_df_text_column(df)
    
    print(f"\nSaving results to {output_csv}")
    save_transcripts(df, output_csv)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process transcripts with case restoration, entity analysis, and summarization")
    parser.add_argument("input_csv", help="Path to input CSV file containing transcripts")
    parser.add_argument("--output", default="processed_transcripts.csv", help="Path to save the processed output")
    
    args = parser.parse_args()
    main(args.input_csv, args.output) 