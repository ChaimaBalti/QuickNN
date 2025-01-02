import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    The CNNRegression class implements a convolutional neural network (CNN) designed for regression tasks. 
    Its primary role is to predict a continuous value (e.g., age estimation, house price prediction, 
    or image-based numerical measurements) based on image inputs. The architecture combines convolutional 
    layers for feature extraction and fully connected layers for regression, incorporating dropout for regularization.
'''

# Define the CNNRegressor class, inheriting from nn.Module
class CNNRegressor(nn.Module):
    def __init__(self):
        """
        Initializes the CNNRegression model.
        
        Architecture:
        - Convolutional layers for feature extraction.
        - Max pooling to downsample features.
        - Fully connected layers for regression.
        - Dropout for regularization.
        """
        super(CNNRegressor, self).__init__()
        
        # Define the first convolutional layer: input channels=3 (e.g., RGB), output channels=32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        # Define the second convolutional layer: input channels=32, output channels=64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Define the third convolutional layer: input channels=64, output channels=128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Define a max pooling layer to downsample feature maps by a factor of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layer: input size matches flattened feature map dimensions (128 * 4 * 4), output=512
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        
        # Fully connected layer for final regression output: 512 to a single continuous value
        self.fc2 = nn.Linear(512, 1)
        
        # Dropout layer with a 30% dropout rate for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Defines the forward pass of the CNNRegression model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            torch.Tensor: Predicted continuous value of shape (batch_size, 1).
        """
        # Apply first convolutional layer, activation function, and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply second convolutional layer, activation function, and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Apply third convolutional layer, activation function, and max pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the feature maps into a vector for the fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        
        # Apply the first fully connected layer with dropout and activation function
        x = self.dropout(F.relu(self.fc1(x)))
        
        # Apply the final fully connected layer for regression
        x = self.fc2(x)
        
        return x

# Example usage of the CNNRegression model
if __name__ == "__main__":
    # Instantiate the model
    model = CNNRegressor()
    
    # Create a sample input tensor (batch_size=1, channels=3, height=32, width=32)
    sample_input = torch.randn(1, 3, 32, 32)
    
    # Perform a forward pass through the model
    output = model(sample_input)
