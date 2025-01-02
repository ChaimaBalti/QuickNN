import torch
import torch.nn as nn

# Define the RNNForecaster class, inheriting from nn.Module
'''
    This function implements a recurrent neural network (RNN) model using LSTM layers for time series forecasting.
    Its primary role is to process sequential data (e.g., historical time series data) and predict future values. 
    By leveraging the LSTM's ability to capture temporal dependencies, the model is well-suited for forecasting tasks such as stock prices, 
    weather conditions, or energy consumption.
'''
class RNNForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        """
        Initializes the RNNForecaster.

        Args:
            input_size (int): Number of features in the input sequence.
            hidden_size (int, optional): Number of features in the hidden state. Defaults to 64.
            num_layers (int, optional): Number of LSTM layers. Defaults to 2.
            output_size (int, optional): Number of features in the output. Defaults to 1.
        """
        super(RNNForecaster, self).__init__()
        
        # Store LSTM configuration
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define an LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,  # Number of input features
            hidden_size=hidden_size,  # Number of features in the hidden state
            num_layers=num_layers,  # Number of stacked LSTM layers
            batch_first=True,  # Input and output tensors are (batch_size, seq_len, feature_size)
            dropout=0.2  # Dropout for regularization
        )
        
        # Fully connected layer for output transformation
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Get the batch size from the input
        batch_size = x.size(0)
        
        # Initialize hidden state and cell state for the LSTM
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)  # Hidden state
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)  # Cell state
        
        # Pass input through the LSTM layer
        # out: tensor of shape (batch_size, seq_len, hidden_size)
        # _: tuple of the final hidden state and cell state
        out, _ = self.lstm(x, (h0, c0))
        
        # Use only the last time step's output for prediction
        # out[:, -1, :] extracts the last time step's features (shape: batch_size, hidden_size)
        out = self.fc(out[:, -1, :])  # Pass through the fully connected layer
        
        return out

# Example usage
if __name__ == "__main__":
    # Define the model with input size 3 (number of features in the sequence)
    model = RNNForecaster(input_size=3)
    
    # Generate a sample batch of input data
    # Shape: (batch_size=32, seq_len=10, features=3)
    sample_input = torch.randn(32, 10, 3)
    
    # Perform a forward pass to get the model's output
    # Output shape: (batch_size=32, output_size=1)
    output = model(sample_input)
