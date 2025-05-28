from transformers import pipeline
import torch

class TextSummarizer:
    def __init__(self, model_name="google/flan-t5-base"):
        """
        Initialize the text summarizer with FLAN-T5 model.
        
        Args:
            model_name (str): Name of the pretrained model to use
        """
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        self.summarizer = pipeline("summarization", 
                                 model=model_name, 
                                 device=self.device)
    
    def generate_summaries(self, text):
        """
        Generate both short and long summaries for a single text.
        
        Args:
            text (str): Input text to summarize
            
        Returns:
            tuple: (short_summary, long_summary)
        """
        input_len = len(text.split())
        
        # Generate summaries of different lengths
        short_len = max(30, min(input_len // 4, 50))
        long_len = max(50, min(input_len // 2, 150))
        
        try:
            short_summary = self.summarizer(text, 
                                          max_length=short_len, 
                                          min_length=20, 
                                          do_sample=False)[0]['summary_text']
            
            long_summary = self.summarizer(text, 
                                         max_length=long_len, 
                                         min_length=40, 
                                         do_sample=False)[0]['summary_text']
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "", ""
        
        return short_summary, long_summary
    
    def summarize_df_text_column(self, df, text_col="restored_text"):
        """
        Generate summaries for a column of text in a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_col (str): Name of the column containing text to summarize
            
        Returns:
            pd.DataFrame: DataFrame with summary columns added
        """
        df = df.copy()
        
        # Generate summaries for each text
        summaries = [self.generate_summaries(text) for text in df[text_col]]
        
        # Add summary columns
        df["short_summary"] = [s[0] for s in summaries]
        df["long_summary"] = [s[1] for s in summaries]
        
        return df 