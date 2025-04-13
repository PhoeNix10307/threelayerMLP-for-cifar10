import os
import numpy as np
import json
from load import get_cifar10_data
from model import ThreeLayerNet
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_model(model_path, hidden_size, activation='relu'):
    """Load trained model from .npz file."""
    params = np.load(model_path, allow_pickle=False)
    model = ThreeLayerNet(input_size=32*32*3, hidden_size=hidden_size, output_size=10, activation=activation)
    model.W1, model.b1 = params['W1'], params['b1']
    model.W2, model.b2 = params['W2'], params['b2']
    return model

def evaluate_model(model, X_test, y_test, batch_size=200):
    """Evaluate model performance on the test set."""
    num_test = X_test.shape[0]
    total_loss, correct_predictions = 0.0, 0
    all_preds = []

    for i in range(0, num_test, batch_size):
        X_batch = X_test[i:i+batch_size]
        y_batch = y_test[i:i+batch_size]

        probs = model.forward(X_batch)
        log_probs = -np.log(np.clip(probs[np.arange(len(y_batch)), y_batch], 1e-10, 1.0))
        total_loss += np.sum(log_probs)

        predictions = np.argmax(probs, axis=1)
        correct_predictions += np.sum(predictions == y_batch)
        all_preds.extend(predictions)

    test_loss = total_loss / num_test
    test_acc = correct_predictions / num_test
    return test_acc, test_loss, np.array(all_preds)

def test_model_from_path(model_path, hidden_size=512, activation='relu'):
    """Test a model from a specified .npz file path."""
    data = get_cifar10_data()
    X_test, y_test = data['X_test'], data['y_test']
    label_names = data['label_names']

    if not os.path.exists(model_path):
        print(f"[Error] Model not found: {model_path}")
        return

    print(f"\nTesting model: {model_path}")

    model = load_model(model_path, hidden_size, activation)
    test_acc, test_loss, predictions = evaluate_model(model, X_test, y_test)

    print("Test Results:")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}")

    print("\nClass-wise Accuracy:")
    print("{:<15} {:<10} {:<10}".format("Class", "Accuracy", "Samples"))
    print("-" * 35)
    for i, name in enumerate(label_names):
        mask = (y_test == i)
        total = np.sum(mask)
        if total == 0:
            acc = 'N/A'
        else:
            acc = np.mean(predictions[mask] == y_test[mask])
            acc = f"{acc:.4f}"
        print("{:<15} {:<10} {:<10}".format(name, acc, total))

    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    save_path = os.path.splitext(model_path)[0] + "_confusion_matrix.png"
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved to {save_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test a model by specifying the model path')
    parser.add_argument('--model', type=str, required=True, help='Path to .npz model file')
    parser.add_argument('--hs', type=int, default=512, help='Hidden size (default: 512)')
    parser.add_argument('--act', type=str, default='relu', help='Activation function (default: relu)')
    args = parser.parse_args()

    test_model_from_path(args.model, args.hs, args.act)
