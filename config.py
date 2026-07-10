"""Hyperparameters and paths — edit here to tune the run."""

DATA_DIR   = './data'
OUTPUT_DIR = './output'

# Training
EPOCHS      = 10
BATCH_SIZE  = 128
LR          = 0.001
WEIGHT_DECAY = 1e-4

# Model
NUM_CLASSES = 10   # CIFAR-10

# Reproducibility
SEED = 42
