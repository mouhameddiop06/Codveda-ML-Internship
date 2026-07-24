# Level 3 · Task 3 — Neural Networks with TensorFlow/Keras

**Objective:** Build a simple feed-forward neural network for a classification task
(handwritten-digit recognition).

## Steps covered
- Load and preprocess the dataset.
- Design a neural network architecture (input, hidden, output layers).
- Train the model using backpropagation.
- Evaluate with accuracy and visualize the training/validation loss.

## Dataset
`sklearn.datasets.load_digits` — 1,797 grayscale 8×8 digit images (10 classes, 0–9).
Loaded offline via scikit-learn, so no internet download is required.
(The same code works for full MNIST via `keras.datasets.mnist`.)

## Run
```bash
python neural_network.py     # script version
# or open neural_network.ipynb
```

## Tools
Python, TensorFlow/Keras, scikit-learn, matplotlib.
