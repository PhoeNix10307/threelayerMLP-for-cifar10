import numpy as np
import os
import json
import time
from itertools import product
from train import train_model, default_config
from model import ThreeLayerNet

class HyperparameterSearch:
    def __init__(self, output_dir='hyperparameter_search'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.search_space = {
            'learning_rate': [1e-2, 5e-3, 1e-3, 5e-4],
            'hidden_size': [64, 128, 256, 512, 1024],
            'reg_lambda': [0.0001, 0.001, 0.01]
        }

        self.fixed_activation = 'relu'

        self.results = []
        self.best_config = None
        self.best_val_acc = 0.0

    def generate_configs(self):
        keys = self.search_space.keys()
        values = self.search_space.values()
        for combo in product(*values):
            yield dict(zip(keys, combo))

    def run_search(self):
        configs = list(self.generate_configs())
        print(f"Starting hyperparameter search with {len(configs)} combinations...")
        total_start_time = time.time()

        for i, params in enumerate(configs):
            config_dir = os.path.join(
                self.output_dir,
                f"lr{params['learning_rate']:.0e}_hs{params['hidden_size']}_reg{params['reg_lambda']:.0e}"
            )
            os.makedirs(config_dir, exist_ok=True)

            flag_path = os.path.join(config_dir, 'finished.flag')
            if os.path.exists(flag_path):
                print(f"[Skipped] Already finished: {config_dir}")
                continue

            config = default_config()
            config.update(params)
            config['activation'] = self.fixed_activation
            config['output_dir'] = config_dir

            print(f"\n[{i+1}/{len(configs)}] Training config: "
                  f"lr={params['learning_rate']:.0e}, "
                  f"hidden_size={params['hidden_size']}, "
                  f"reg_lambda={params['reg_lambda']}")

            start_time = time.time()
            model, metrics = train_model(config)
            train_time = time.time() - start_time

            # Save per-config training time
            time_path = os.path.join(config_dir, 'train_time.txt')
            with open(time_path, 'w') as f:
                f.write(f"Training time (s): {train_time:.2f}\n")

            # Save training history
            history_path = os.path.join(config_dir, 'training_history.npz')
            np.savez(history_path,
                     train_loss=metrics['train_loss_history'],
                     val_loss=metrics['val_loss_history'],
                     val_acc=metrics['val_acc_history'])

            # Save final model
            final_model_path = os.path.join(config_dir, 'final_model.npz')
            np.savez(final_model_path,
                     W1=model.W1, b1=model.b1,
                     W2=model.W2, b2=model.b2)

            # Save result
            result = {
                'config': config.copy(),
                'train_time': train_time,
                'train_time_path': time_path,
                'best_val_acc': metrics['best_val_acc'],
                'best_epoch': metrics['best_epoch'],
                'training_curves': {
                    'train_loss': metrics['train_loss_history'],
                    'val_loss': metrics['val_loss_history'],
                    'val_acc': metrics['val_acc_history']
                },
                'training_history_path': history_path,
                'final_model_path': final_model_path
            }

            self.results.append(result)

            if metrics['best_val_acc'] > self.best_val_acc:
                self.best_val_acc = metrics['best_val_acc']
                self.best_config = config.copy()
                self.best_config['best_training_curves'] = result['training_curves']
                print(f"→ New best validation accuracy: {self.best_val_acc:.4f}")

            self.save_results()

            # Mark as finished
            with open(flag_path, 'w') as f:
                f.write('done')

        # Save total training time
        total_time = time.time() - total_start_time
        avg_time = total_time / len(self.results) if self.results else 0
        print(f"\nHyperparameter search completed.")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per config: {avg_time:.2f} seconds")
        with open(os.path.join(self.output_dir, 'total_time.txt'), 'w') as f:
            f.write(f"Total time (s): {total_time:.2f}\n")
            f.write(f"Average time per config (s): {avg_time:.2f}\n")

    def save_results(self):
        results_path = os.path.join(self.output_dir, 'search_results.json')
        existing_configs = set()

        # Load existing results if available
        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                previous_results = data.get('results', [])
                for res in previous_results:
                    key = json.dumps(res['config'], sort_keys=True)
                    existing_configs.add(key)
                self.results = previous_results + [
                    r for r in self.results
                    if json.dumps(r['config'], sort_keys=True) not in existing_configs
                ]
                # Preserve best result across sessions
                if data.get('best_val_acc', 0) > self.best_val_acc:
                    self.best_val_acc = data['best_val_acc']
                    self.best_config = data['best_config']

        # Update best config if any result this round is better
        for r in self.results:
            if r['best_val_acc'] > self.best_val_acc:
                self.best_val_acc = r['best_val_acc']
                self.best_config = r['config']
                self.best_config['best_training_curves'] = r['training_curves']

        # Save updated results
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'results': self.results,
                'best_config': self.best_config,
                'best_val_acc': self.best_val_acc
            }, f, indent=2)

        # Save best config separately
        if self.best_config:
            with open(os.path.join(self.output_dir, 'best_config.json'), 'w', encoding='utf-8') as f:
                json.dump(self.best_config, f, indent=2)


    def analyze_results(self):
        """Analyze and print results from search_results.json, even if not run in the same session."""
        result_path = os.path.join(self.output_dir, 'search_results.json')
        if not os.path.exists(result_path):
            print("No results file found. Please run the search first.")
            return

        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.results = data.get('results', [])
            self.best_config = data.get('best_config', {})
            self.best_val_acc = data.get('best_val_acc', 0.0)

        if not self.results:
            print("No results to analyze.")
            return

        print("\n=== Search Summary ===")
        print(f"Total configurations tested: {len(self.results)}")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print("Best configuration:")
        for k, v in self.best_config.items():
            if k in self.search_space:
                print(f"  {k}: {v}")

        sorted_results = sorted(self.results, key=lambda x: x['best_val_acc'], reverse=True)

        print("\nTop-5 configurations:")
        for i, res in enumerate(sorted_results[:5]):
            print(f"{i+1}. val_acc={res['best_val_acc']:.4f} | "
                  f"lr={res['config']['learning_rate']:.0e}, "
                  f"hs={res['config']['hidden_size']}, "
                  f"reg={res['config']['reg_lambda']}")

        analysis_path = os.path.join(self.output_dir, 'search_analysis.txt')
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write("=== Hyperparameter Search Summary ===\n")
            f.write(f"Total configurations tested: {len(self.results)}\n")
            f.write(f"Best validation accuracy: {self.best_val_acc:.4f}\n")
            f.write("Best configuration:\n")
            for k, v in self.best_config.items():
                if k in self.search_space:
                    f.write(f"  {k}: {v}\n")

            f.write("\nTop-5 configurations:\n")
            for i, res in enumerate(sorted_results[:5]):
                f.write(f"{i+1}. val_acc={res['best_val_acc']:.4f} | "
                        f"lr={res['config']['learning_rate']:.0e}, "
                        f"hs={res['config']['hidden_size']}, "
                        f"reg={res['config']['reg_lambda']}\n")


def main():
    searcher = HyperparameterSearch(output_dir='hyperparameter_search')
    searcher.run_search()
    searcher.analyze_results()

if __name__ == '__main__':
    main()
