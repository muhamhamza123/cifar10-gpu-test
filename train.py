"""Training script — run directly or called from the notebook."""

import os
import sys
import time
import torch
import torch.nn as nn

import config
from model   import CIFAR10CNN
from dataset import get_loaders
from utils   import accuracy, save_checkpoint, save_history, plot_history


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_acc = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc  += accuracy(outputs, targets)
    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_acc = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        total_loss += criterion(outputs, targets).item()
        total_acc  += accuracy(outputs, targets)
    n = len(loader)
    return total_loss / n, total_acc / n


def main():
    torch.manual_seed(config.SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')
    if device.type == 'cuda':
        print(f'[gpu] {torch.cuda.get_device_name(0)}')
        print(f'[gpu] memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print('[data] loading CIFAR-10…')
    train_loader, val_loader = get_loaders(config.DATA_DIR, config.BATCH_SIZE)
    print(f'[data] {len(train_loader.dataset)} train / {len(val_loader.dataset)} val')

    model     = CIFAR10CNN(config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'[model] parameters: {total_params:,}')

    history   = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc  = 0.0
    ckpt_path = os.path.join(config.OUTPUT_DIR, 'best_model.pt')

    print(f'\n{"Epoch":>6}  {"Train Loss":>10}  {"Train Acc":>9}  {"Val Loss":>8}  {"Val Acc":>7}  {"LR":>8}  {"Time":>6}')
    print('-' * 72)

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_epoch(model,  val_loader,   criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)

        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        print(f'{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  {va_loss:>8.4f}  {va_acc:>7.4f}  {lr:>8.6f}  {elapsed:>5.1f}s')
        sys.stdout.flush()

        if va_acc > best_acc:
            best_acc = va_acc
            save_checkpoint({'epoch': epoch, 'model': model.state_dict(),
                             'optimizer': optimizer.state_dict(), 'best_acc': best_acc}, ckpt_path)

    save_history(history, os.path.join(config.OUTPUT_DIR, 'history.json'))
    plot_history(history, config.OUTPUT_DIR)
    print(f'\n[done] best val accuracy: {best_acc:.4f}')
    print(f'[done] outputs saved to {config.OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
