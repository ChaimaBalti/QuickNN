# QuickNN: A Simplified Neural Network Library

QuickNN is a lightweight Python library designed to simplify the process of building, testing, and deploying neural network architectures. Whether you're a beginner or an experienced machine learning engineer, QuickNN provides intuitive tools and predefined models to accelerate your AI workflows.

## Features
- **Predefined Architectures**: Access ready-to-use neural network models, such as CNN classifiers.
- **Testing Utilities**: Built-in functions to validate and benchmark models.
- **User-Friendly Design**: Streamlined codebase with minimal dependencies.

## Installation
To install QuickNN, clone the repository and run the following command:
```bash
pip install -e .
```

## Requirements
QuickNN requires Python 3.8 or higher and the following dependencies:
- `torch`

You can also install all dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Quick Start
Here’s a simple example of using a CNN classifier from QuickNN:

```python
from QuickNN.Test_Models import CNNClassifier
import torch

# Define model
model = CNNClassifier(input_channels=3, num_classes=10)

# Dummy input
dummy_input = torch.randn(8, 3, 32, 32)  # Batch size: 8, Channels: 3, Image size: 32x32
output = model(dummy_input)

print("Output shape:", output.shape)
```

## Directory Structure
```
QuickNN/
├── __init__.py
├── Test_Models.py
└── utils.py
setup.py
requirements.txt
README.md
```

## Contributing
Contributions are welcome! If you have ideas for new features or improvements, feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
```

---

### **`requirements.txt`**
```plaintext
torch>=1.10.0
```

---
