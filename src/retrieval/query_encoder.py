from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class CodeBERTEncoder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.model.eval()
    
    def encode(self, text: str) -> np.ndarray:
        """Encode query text using CodeBERT (same as document embeddings)."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # CLS token embedding - must match how Nicholas generated embeddings
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()