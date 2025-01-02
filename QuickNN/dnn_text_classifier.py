import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the DNNTextClassifier class, inheriting from nn.Module
''' 
    This function defines a deep neural network (DNN) model for text classification. 
    Its purpose is to classify text sequences into a predefined number of categories. 
    The model uses an embedding layer to represent words as dense vectors, 
    followed by several fully connected layers with activation functions 
    and dropout for learning hierarchical features and reducing overfitting. 
    It is ideal for tasks like sentiment analysis, topic categorization, or spam detection.
'''

class DNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=100):
        """
        Initializes the DNNTextClassifier.

        Args:
            vocab_size (int): The size of the vocabulary.
            num_classes (int): The number of output classes for classification.
            embedding_dim (int, optional): The dimension of the word embeddings. Defaults to 100.
        """
        super(DNNTextClassifier, self).__init__()
        
        # Define the embedding layer to convert input tokens into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Define a sequential feed-forward neural network with multiple layers
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 256),  # Linear transformation: embedding_dim -> 256
            nn.ReLU(),                     # Activation function: ReLU
            nn.Dropout(0.3),               # Dropout layer with 30% probability to prevent overfitting
            nn.Linear(256, 128),           # Linear transformation: 256 -> 128
            nn.ReLU(),                     # Activation function: ReLU
            nn.Dropout(0.3),               # Dropout layer with 30% probability
            nn.Linear(128, 64),            # Linear transformation: 128 -> 64
            nn.ReLU(),                     # Activation function: ReLU
            nn.Linear(64, num_classes)     # Final linear layer: 64 -> num_classes
        )
        
    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Convert token IDs to embeddings and take the mean across sequence length
        embedded = self.embedding(x).mean(dim=1)
        
        # Pass the processed embeddings through the feed-forward layers
        return self.layers(embedded)

# Example usage
if __name__ == "__main__":
    # Define hyperparameters
    vocab_size = 10000  # Vocabulary size
    num_classes = 5     # Number of output classes
    
    # Instantiate the model
    model = DNNTextClassifier(vocab_size, num_classes)
    
    # Generate a sample batch of input data (batch_size=32, sequence_length=100)
    sample_input = torch.randint(0, vocab_size, (32, 100))
    
    # Perform a forward pass to obtain the model's predictions
    output = model(sample_input)
