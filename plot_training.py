import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_config_dirs(search_dir):
    """Return list of config result directories under search_dir."""
    return [
        os.path.join(search_dir, d) for d in os.listdir(search_dir)
        if os.path.isdir(os.path.join(search_dir, d)) and os.path.exists(os.path.join(search_dir, d, 'training_history.npz'))
    ]

def plot_single_config(dir_path):
    """Plot and save training curves for one configuration."""
    history_path = os.path.join(dir_path, 'training_history.npz')
    history = np.load(history_path)
    
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    val_acc = history['val_acc']

    plt.figure(figsize=(15, 5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss', linestyle='--', color='blue')
    plt.plot(val_loss, label='Validation Loss', linestyle='-', color='green')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(val_acc, label='Validation Accuracy', color='red')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy Curve')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(dir_path, 'training_curves.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot to {save_path}")

def plot_all_configs(search_dir='hyperparameter_search'):
    config_dirs = load_config_dirs(search_dir)
    print(f"Found {len(config_dirs)} configurations.")
    for config_dir in config_dirs:
        plot_single_config(config_dir)

if __name__ == '__main__':
    plot_all_configs('hyperparameter_search')
