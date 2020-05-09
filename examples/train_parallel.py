import sys
sys.path.append('..')
from defacing.training.training_parallel import train

for fold in range(1, 11):
    train(fold=fold)
