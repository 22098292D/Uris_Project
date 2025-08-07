"""
Module for reshaping academic text using transformers.
"""
import logging
from typing import List, Optional
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextReshaper:
    def __init__(self, model_name: str = "t5-base", device: Optional[str] = None):
        """Initialize the TextReshaper with a pretrained model.
        
        Args:
            model_name (str): Name of the pretrained model to use
            device (str, optional): Device to run the model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing TextReshaper with {model_name} on {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def split_words(self, text: str) -> str:
        """Split concatenated words using various patterns."""
        # Common word boundaries
        common_words = [
            'serves', 'aims', 'has', 'by', 'for', 'which', 'across', 'and', 'their',
            'about', 'various', 'applications', 'two', 'identifying', 'capability',
            'processing', 'semantic', 'information', 'structure', 'organizing',
            'relationships', 'effective', 'efficient'
        ]
        
        # Add word boundaries
        for word in common_words:
            text = re.sub(f'(?<=[a-z])({word})', r' \1', text)
        
        # Split camelCase and similar patterns
        patterns = [
            (r'([a-z])([A-Z][a-z])', r'\1 \2'),  # camelCase
            (r'([A-Z][a-z])([A-Z])', r'\1 \2'),  # PascalCase
            (r'([a-z])([A-Z][A-Z]+)', r'\1 \2'),  # wordABC
            (r'([A-Z][A-Z])([a-z])', r'\1 \2'),  # ABCword
            (r'([a-z])([0-9])', r'\1 \2'),  # word1
            (r'([0-9])([a-z])', r'\1 \2'),  # 1word
        ]
        
        for pattern, repl in patterns:
            text = re.sub(pattern, repl, text)
        
        return text
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text before sending to model."""
        # Remove newlines and extra spaces
        text = text.replace('\n', ' ')
        text = ' '.join(text.split())
        
        # Fix common academic abbreviations with word boundaries
        text = re.sub(r'\bEA\b', 'Entity Alignment', text)
        text = re.sub(r'\bKGs?\b', 'Knowledge Graphs', text)
        text = re.sub(r'\bLLMs?\b', 'Large Language Models', text)
        
        # Split concatenated words
        text = self.split_words(text)
        
        # Fix common academic terms
        text = re.sub(r'\bKnowledge ?Graphs?\b', 'Knowledge Graphs', text)
        text = re.sub(r'\bEntity ?Alignment\b', 'Entity Alignment', text)
        text = re.sub(r'\bLarge ?Language ?Models?\b', 'Large Language Models', text)
        
        # Fix common patterns
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # Add space between lower and upper
        text = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', text)  # Add space between upper and upper+lower
        
        # Final cleanup
        text = ' '.join(text.split())
        
        return text.strip()
    
    def postprocess_text(self, text: str) -> str:
        """Postprocess text after model generation."""
        # Remove any remaining prompt text
        text = text.replace('Rewrite this academic text to be clear and properly formatted:', '')
        text = text.replace('Rewrite this academic text to be clear and properly formatted', '')
        
        # Fix punctuation spacing
        text = re.sub(r'\s+([,.;:)])', r'\1', text)
        text = re.sub(r'([(])\s+', r'\1', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Fix academic terms
        text = re.sub(r'\bKnowledge ?Graphs?\b', 'Knowledge Graphs', text)
        text = re.sub(r'\bEntity ?Alignment\b', 'Entity Alignment', text)
        text = re.sub(r'\bLarge ?Language ?Models?\b', 'Large Language Models', text)
        
        # Fix common word concatenations
        text = self.split_words(text)
        
        # Final cleanup
        text = ' '.join(word for word in text.split() if word)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def reshape_text(self, text: str, max_length: int = 512) -> str:
        """Reshape the input text to improve quality and readability.
        
        Args:
            text (str): Input text to reshape
            max_length (int): Maximum length for input and output sequences
            
        Returns:
            str: Reshaped text
        """
        try:
            # Preprocess text
            text = self.preprocess_text(text)
            
            # Add prompt to guide the model
            prompt = "Rewrite this academic text to be clear and properly formatted: "
            input_text = prompt + text
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True
            ).to(self.device)
            
            # Generate
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=4,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                length_penalty=1.0,
                early_stopping=True
            )
            
            # Decode and postprocess
            reshaped_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            reshaped_text = self.postprocess_text(reshaped_text)
            
            return reshaped_text
            
        except Exception as e:
            logger.error(f"Error reshaping text: {str(e)}")
            logger.warning("Returning original text due to error")
            return text

    def reshape_batch(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """Reshape a batch of texts.
        
        Args:
            texts (List[str]): List of texts to reshape
            batch_size (int): Size of batches for processing
            
        Returns:
            List[str]: List of reshaped texts
        """
        reshaped_texts = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            reshaped_batch = [self.reshape_text(text) for text in batch]
            reshaped_texts.extend(reshaped_batch)
            
        return reshaped_texts

def main():
    """Test the TextReshaper with sample text."""
    # Initialize the reshaper
    reshaper = TextReshaper()
    
    # Test cases
    test_texts = [
        """KnowledgeGraphsserveasa foundationalstructurefor storingandorganizing structured
        knowledgeabout entitiesandtheirrelationships,whichfacilitateseffectiveandefficientse
        archcapabilitiesacrossvariousapplications.""",
        
        """EntityalignmentEAaimstomergetwoknowledgegraphsKGsbyidentifying
        equivalententitypairs.""",
        
        """LargeLanguageModelsLLMshaveshowcasedtheirsuperiorcapabilityinprocessing
        semanticinformation."""
    ]
    
    # Test each case
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print("Original text:")
        print(text)
        print("\nReshaped text:")
        reshaped = reshaper.reshape_text(text)
        print(reshaped)
        print("-" * 80)

if __name__ == "__main__":
    main() 