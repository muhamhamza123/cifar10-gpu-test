"""Metrics, checkpoint helpers, and plotting."""

import os
import json
import torch
import matplotlib
matplotlib.use('Agg')  # headless — no display on Puhti
import matplotlib.pyplot as plt


def accuracy(outputs, targets):
    _, preds = outputs.max(1)
    return preds.eq(targets).float().mean().item()


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f'[checkpoint] saved → {path}')


def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt.get('epoch', 0), ckpt.get('best_acc', 0.0)


def save_history(history: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)


def plot_history(history: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history['train_loss'], label='Train loss')
    ax1.plot(epochs, history['val_loss'],   label='Val loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss'); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, history['train_acc'], label='Train acc')
    ax2.plot(epochs, history['val_acc'],   label='Val acc')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy'); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(out_dir, 'training_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'[plot] saved → {path}')
