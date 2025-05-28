from punctuators.models import PunctCapSegModelONNX
import pandas as pd

class TextRestorer:
    def __init__(self, model_name="1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"):
        """
        Initialize the text restorer with a punctuation and capitalization model.
        
        Args:
            model_name (str): Name of the pretrained model to use
        """
        self.model = PunctCapSegModelONNX.from_pretrained(model_name)
    
    def restore_text(self, text):
        """
        Restore proper casing and punctuation to a single text.
        
        Args:
            text (str): Input text to restore
            
        Returns:
            str: Restored text with proper casing and punctuation
        """
        text = str(text).lower()
        restored = self.model.infer([text])
        return " ".join(restored[0]) if isinstance(restored[0], list) else str(restored[0])
    
    def restore_df_text_column(self, df, text_col="new_text", output_col="restored_text"):
        """
        Restore proper casing and punctuation to a column of text in a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_col (str): Name of the column containing text to restore
            output_col (str): Name of the column to store restored text
            
        Returns:
            pd.DataFrame: DataFrame with restored text column added
        """
        texts = df[text_col].astype(str).str.lower().tolist()
        restored = self.model.infer(texts)
        flattened = [" ".join(text) if isinstance(text, list) else str(text) for text in restored]
        
        df = df.copy()
        df[output_col] = flattened
        return df 