import os
import pickle
import tarfile
import urllib.request
import numpy as np
from sklearn.model_selection import train_test_split

CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DEFAULT_DATA_DIR = 'data/cifar10'

def download_cifar10(data_dir=DEFAULT_DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, 'cifar-10-python.tar.gz')
    
    if not os.path.exists(file_path):
        print("Downloading CIFAR-10...")
        urllib.request.urlretrieve(CIFAR10_URL, file_path)
        print("Download complete.")
    
    return file_path

def extract_cifar10(file_path):
    data_dir = os.path.dirname(file_path)
    extracted_path = os.path.join(data_dir, 'cifar-10-batches-py')
    
    if not os.path.exists(extracted_path):
        print("Extracting CIFAR-10...")
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=data_dir)
        print("Extraction complete.")
    
    return extracted_path

def load_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    images = batch['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(batch['labels'])
    return images, labels


def preprocess_images(images, mean_image=None):
    images = images.astype(np.float32) / 255.0
    images = images.reshape(images.shape[0], -1)
    if mean_image is not None:
        images -= mean_image
    return images

def load_cifar10(data_dir=DEFAULT_DATA_DIR, validation_size=5000):
    file_path = download_cifar10(data_dir)
    extracted_path = extract_cifar10(file_path)

    X_train, y_train = [], []
    for i in range(1, 6):
        X, y = load_batch(os.path.join(extracted_path, f'data_batch_{i}'))
        X_train.append(X)
        y_train.append(y)

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    X_test, y_test = load_batch(os.path.join(extracted_path, 'test_batch'))

    with open(os.path.join(extracted_path, 'batches.meta'), 'rb') as f:
        label_names = pickle.load(f, encoding='latin1')['label_names']

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_size,
        stratify=y_train, random_state=42
    )

    mean_image = np.mean(X_train.astype(np.float32).reshape(X_train.shape[0], -1) / 255.0, axis=0)

    X_train = preprocess_images(X_train, mean_image)
    X_val = preprocess_images(X_val, mean_image)
    X_test = preprocess_images(X_test, mean_image)

    return X_train, y_train, X_val, y_val, X_test, y_test, label_names

def get_cifar10_data(validation_size=5000):
    X_train, y_train, X_val, y_val, X_test, y_test, label_names = load_cifar10(validation_size=validation_size)
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'label_names': label_names
    }

if __name__ == '__main__':
    data = get_cifar10_data()

    print("\nDataset Summary:")
    print(f"Train: {data['X_train'].shape}, {data['y_train'].shape}")
    print(f"Val:   {data['X_val'].shape}, {data['y_val'].shape}")
    print(f"Test:  {data['X_test'].shape}, {data['y_test'].shape}")
    print("Labels:", data['label_names'])
