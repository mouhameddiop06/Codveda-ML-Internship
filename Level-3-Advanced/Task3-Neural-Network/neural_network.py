"""
Level 3 - Task 3: Neural Networks with TensorFlow/Keras
Codveda Technologies ML Internship

Feed-forward neural network for handwritten-digit classification (load_digits, offline).
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

tf.random.set_seed(42)
np.random.seed(42)


def main():
    # Load & preprocess -------------------------------------------------------
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Architecture ------------------------------------------------------------
    model = keras.Sequential([
        layers.Input(shape=(64,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    # Train -------------------------------------------------------------------
    history = model.fit(X_train, y_train, validation_split=0.15,
                        epochs=40, batch_size=32, verbose=0)

    # Evaluate ----------------------------------------------------------------
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.3f}\n")
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    print(classification_report(y_test, y_pred))

    # Training curves ---------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(history.history["loss"], label="train")
    ax[0].plot(history.history["val_loss"], label="validation")
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(history.history["accuracy"], label="train")
    ax[1].plot(history.history["val_accuracy"], label="validation")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig("output_training_curves.png", dpi=110)
    print("Saved training curves.")


if __name__ == "__main__":
    main()
