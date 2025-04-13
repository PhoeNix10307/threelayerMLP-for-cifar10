This repository implements a simple three-layer MLP classifier for the CIFAR-10 dataset using NumPy only, with modular support for training, hyperparameter search, evaluation, and visual analysis of learned parameters.

# File Structure
``` text
.
├── best_model/                # Place best_model.npz here (Suggested. 
│                                test.py and para_visualize.ipynb will
│                                produce several pngs at the same
│                                directory of the model, so you may
│                                collect them in a folder)
├── data/cifar10/              # Place CIFAR-10 dataset here
├── hyperparameter_search/     # Archive for the results of training
├── load.py                    # Load and preprocess CIFAR-10 from data/
├── model.py                   # MLP model (ThreeLayerNet)
├── train.py                   # Train a single configuration
├── search.py                  # Run hyperparameter search
├── test.py                    # Test a specific config
├── plot_training.py           # Plot training curves
└── plot_visualize.ipynb       # Visualize a model
```

# Training and testing procedure

## load.py: Load and preprocess the dataset
Place CIFAR-10 dataset in data/ ("data/cifar10/cifar-10-python.tar.gz") and then:
```bash
python load.py
```

## train.py: Train a model using specific hyperparameters
The following arguments can be passed to `train.py` to configure the model architecture and training strategy:

| Argument        | Type    | Description                                                     | Default  |
| --------------- | ------- | --------------------------------------------------------------- | -------- |
| `--lr`          | `float` | Initial learning rate                                           | `1e-3`   |
| `--hs`          | `int`   | Number of hidden units in the hidden layer                      | `512`    |
| `--reg`         | `float` | L2 regularization strength                                      | `0.01`   |
| `--act`         | `str`   | Activation function (`relu`, `tanh`, `sigmoid`, etc.)           | `'relu'` |
| `--mom`         | `float` | Momentum factor for SGD                                         | `0.9`    |
| `--batch_size`  | `int`   | Mini-batch size                                                 | `200`    |
| `--epochs`      | `int`   | Number of training epochs                                       | `100`    |
| `--lr_decay`    | `float` | Learning rate decay factor                                      | `0.95`   |
| `--decay_epoch` | `int`   | Decay the learning rate every N epochs                          | `5`      |
| `--val_size`    | `int`   | Size of the validation set (sampled from training data)         | `5000`   |
| `--save_result` | `flag`  | If set, saves the result incrementally to `search_results.json` | Disabled |

---

Example Usage:
```bash
python train.py --lr 0.005 --hs 1024 --reg 0.001 --act relu --epochs 80 --save_result
```


## search.py: Search for the best configuration of lr, hs and reg
Set the configuration search space in search.py and then:
```bash
python search.py
```

## test.py: Test a specific model 
Set directory to the model, the hidden size, and the activation function. The hidden size and the activation function must be identical to the model.

Example Usage:
```bash
python test.py --model best_model/best_model.npz --hs 1024 --act relu
```

## plot_training.py: Plot training curves
```bash
python plot_training.py
```
This will add the training plots respectively for each configuration in hyperparameter_search/. 
**Note: You must have trained a model by train.py or search.py, and then the training history is saved to hyperparameter_search/, so that you can plot the training curves.**

## para_visualize.ipynb: Visualize a model
Set the model path and visualize the model.