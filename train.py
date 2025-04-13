import numpy as np
import os
import time
import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from load import get_cifar10_data
from model import ThreeLayerNet

def default_config():
    return {
        'hidden_size': 512,
        'activation': 'relu',
        'learning_rate': 1e-3,
        'momentum': 0.9,
        'batch_size': 200,
        'epochs': 100,
        'reg_lambda': 0.01,
        'lr_decay': 0.95,
        'decay_epoch': 5,
        'validation_size': 5000,
        'output_dir': None
    }

def SGDOptimizer(learning_rate, momentum, lr_decay, decay_epoch):
    class Optimizer:
        def __init__(self):
            self.learning_rate = learning_rate
            self.momentum = momentum
            self.lr_decay = lr_decay
            self.decay_epoch = decay_epoch
            self.velocity = {}

        def step(self, params, grads):
            if not self.velocity:
                self.velocity = {k: np.zeros_like(v) for k, v in params.items()}
            for k in params:
                self.velocity[k] = self.momentum * self.velocity[k] - self.learning_rate * grads[k]
                params[k] += self.velocity[k]

        def update_learning_rate(self, epoch):
            if epoch > 0 and epoch % self.decay_epoch == 0:
                self.learning_rate *= self.lr_decay
                print(f"Epoch {epoch}: Learning rate decayed to {self.learning_rate:.6f}")

    return Optimizer()

def plot_training_curves(train_loss, val_loss, val_acc, save_dir):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='green')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(val_acc, label='Validation Accuracy', color='red')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Training curves saved to {path}")

def train_model(config, save_result=False):
    total_start = time.time()

    if not config.get('output_dir'):
        name = f"lr{config['learning_rate']:.0e}_hs{config['hidden_size']}_reg{config['reg_lambda']:.0e}"
        config['output_dir'] = os.path.join('hyperparameter_search', name)
    os.makedirs(config['output_dir'], exist_ok=True)

    data = get_cifar10_data(config['validation_size'])
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']

    model = ThreeLayerNet(32*32*3, config['hidden_size'], 10, config['activation'])
    optimizer = SGDOptimizer(config['learning_rate'], config['momentum'], config['lr_decay'], config['decay_epoch'])

    num_train = X_train.shape[0]
    iters_per_epoch = max(num_train // config['batch_size'], 1)

    train_loss_history = [model.loss(X_train, y_train, config['reg_lambda'])]
    val_loss_history = [model.loss(X_val, y_val, config['reg_lambda'])]
    val_acc_history = [np.mean(np.argmax(model.forward(X_val), axis=1) == y_val)]

    best_val_acc = val_acc_history[0]
    best_epoch = 0
    best_params = {k: v.copy() for k, v in model.__dict__.items() if isinstance(v, np.ndarray)}

    print(f"Epoch   0 | Train Loss: {train_loss_history[0]:.4f} | "
          f"Val Loss: {val_loss_history[0]:.4f} | Val Acc: {val_acc_history[0]:.4f}")

    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()
        optimizer.update_learning_rate(epoch)

        indices = np.random.permutation(num_train)
        X_train, y_train = X_train[indices], y_train[indices]
        epoch_loss = 0.0

        for i in range(iters_per_epoch):
            start, end = i * config['batch_size'], (i + 1) * config['batch_size']
            X_batch, y_batch = X_train[start:end], y_train[start:end]

            epoch_loss += model.loss(X_batch, y_batch, config['reg_lambda'])
            grads = dict(zip(['W1', 'b1', 'W2', 'b2'], model.backward(X_batch, y_batch, config['reg_lambda'])))
            params = dict(W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)
            optimizer.step(params, grads)

        train_loss_history.append(epoch_loss / iters_per_epoch)
        val_loss = model.loss(X_val, y_val, config['reg_lambda'])
        val_acc = np.mean(np.argmax(model.forward(X_val), axis=1) == y_val)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_params = {k: v.copy() for k, v in model.__dict__.items() if isinstance(v, np.ndarray)}
            np.savez(os.path.join(config['output_dir'], 'best_model.npz'), **best_params)
            print(f"Epoch {epoch}: New best val acc {val_acc:.4f}, model saved.")

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss_history[-1]:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {time.time() - epoch_start:.2f}s")

    for k in best_params:
        setattr(model, k, best_params[k])

    plot_training_curves(train_loss_history, val_loss_history, val_acc_history, config['output_dir'])
    np.savez(os.path.join(config['output_dir'], 'training_history.npz'),
             train_loss=train_loss_history, val_loss=val_loss_history, val_acc=val_acc_history)

    train_time = time.time() - total_start
    with open(os.path.join(config['output_dir'], 'train_time.txt'), 'w') as f:
        f.write(f"Training time (s): {train_time:.2f}\n")

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f} @ Epoch {best_epoch}")
    print(f"Total training time: {train_time:.2f}s")

    if save_result:
        results_path = os.path.join('hyperparameter_search', 'search_results.json')
        os.makedirs('hyperparameter_search', exist_ok=True)

        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results = data.get('results', [])
                existing_keys = {json.dumps(r['config'], sort_keys=True) for r in results}
                best_val_record = data.get('best_val_acc', 0.0)
                best_config_record = data.get('best_config', {})
        else:
            results, existing_keys = [], set()
            best_val_record, best_config_record = 0.0, {}

        config_key = json.dumps(config, sort_keys=True)
        if config_key not in existing_keys:
            result = {
                'config': config.copy(),
                'train_time': train_time,
                'train_time_path': os.path.join(config['output_dir'], 'train_time.txt'),
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch,
                'training_curves': {
                    'train_loss': train_loss_history,
                    'val_loss': val_loss_history,
                    'val_acc': val_acc_history
                },
                'training_history_path': os.path.join(config['output_dir'], 'training_history.npz'),
                'final_model_path': os.path.join(config['output_dir'], 'best_model.npz')
            }
            results.append(result)
            if best_val_acc > best_val_record:
                best_val_record = best_val_acc
                best_config_record = config.copy()
                best_config_record['best_training_curves'] = result['training_curves']

            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'results': results,
                    'best_config': best_config_record,
                    'best_val_acc': best_val_record
                }, f, indent=2)

            with open(os.path.join('hyperparameter_search', 'best_config.json'), 'w', encoding='utf-8') as f:
                json.dump(best_config_record, f, indent=2)

    return model, {
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'val_acc_history': val_acc_history,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--hs', type=int, help='Hidden size')
    parser.add_argument('--reg', type=float, help='Regularization lambda')
    parser.add_argument('--act', type=str, help='Activation function')
    parser.add_argument('--mom', type=float, help='Momentum')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr_decay', type=float, help='Learning rate decay')
    parser.add_argument('--decay_epoch', type=int, help='Decay every N epochs')
    parser.add_argument('--val_size', type=int, help='Validation set size')
    parser.add_argument('--save_result', action='store_true', help='Whether to save result to search_results.json')
    args = parser.parse_args()

    cfg = default_config()
    if args.lr: cfg['learning_rate'] = args.lr
    if args.hs: cfg['hidden_size'] = args.hs
    if args.reg: cfg['reg_lambda'] = args.reg
    if args.act: cfg['activation'] = args.act
    if args.mom: cfg['momentum'] = args.mom
    if args.batch_size: cfg['batch_size'] = args.batch_size
    if args.epochs: cfg['epochs'] = args.epochs
    if args.lr_decay: cfg['lr_decay'] = args.lr_decay
    if args.decay_epoch: cfg['decay_epoch'] = args.decay_epoch
    if args.val_size: cfg['validation_size'] = args.val_size

    model, metrics = train_model(cfg, save_result=args.save_result)
