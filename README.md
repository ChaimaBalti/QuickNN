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
QuickNN requires the following libraries to function properly:
- `torch`: PyTorch library for building and training neural networks.
- `torchvision`: For handling datasets and transforms.
- `matplotlib`: Used for plotting and visualizations.
- `tqdm`: Provides progress bars for training and data processing.
- `scikit-learn`: For data preprocessing and evaluation metrics.

You can also install all dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Directory Structure
```
QuickNN/
├── __init__.py
├── cnn_classifier.py
├── cnn_regressor.py
├── dnn_text_classifier.py
└── rnn_forecaster.py
Testing_QuickNN.ipynb
requirements.txt
README.md
```

## Contributing
Contributions are welcome! If you have ideas for new features or improvements, feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

