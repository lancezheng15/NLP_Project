import spacy
import pandas as pd
from collections import Counter
import os

class EntityAnalyzer:
    def __init__(self, model_name="en_core_web_sm"):
        """
        Initialize the entity analyzer with a spaCy model.
        
        Args:
            model_name (str): Name of the spaCy model to use
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            os.system(f"python -m spacy download {model_name}")
            self.nlp = spacy.load(model_name)
    
    def extract_entities(self, text):
        """
        Extract named entities from a single text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            list: List of tuples containing (entity_text, entity_label)
        """
        doc = self.nlp(str(text))
        return [(ent.text, ent.label_) for ent in doc.ents]
    
    def analyze_df_text_column(self, df, text_col="restored_text", output_col="entities"):
        """
        Extract named entities from a column of text in a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_col (str): Name of the column containing text to analyze
            output_col (str): Name of the column to store extracted entities
            
        Returns:
            pd.DataFrame: DataFrame with entities column added
        """
        df = df.copy()
        df[output_col] = df[text_col].apply(self.extract_entities)
        return df
    
    def get_entity_statistics(self, df, entities_col="entities"):
        """
        Calculate statistics about the entities in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame with entities column
            entities_col (str): Name of the column containing entities
            
        Returns:
            dict: Dictionary containing entity statistics
        """
        # Flatten all entities
        all_entities = []
        for entity_list in df[entities_col]:
            all_entities.extend(entity_list)
        
        # Count entity types
        entity_types = Counter(label for _, label in all_entities)
        
        # Count unique entities per type
        unique_entities_by_type = {}
        for text, label in all_entities:
            if label not in unique_entities_by_type:
                unique_entities_by_type[label] = set()
            unique_entities_by_type[label].add(text.lower())
        
        unique_counts = {label: len(entities) for label, entities in unique_entities_by_type.items()}
        
        return {
            "total_entities": len(all_entities),
            "entity_type_counts": dict(entity_types),
            "unique_entities_per_type": unique_counts
        } 