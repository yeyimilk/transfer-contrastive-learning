from models import CNNModel, MLP, LSTM
from trainer_utils import train_n_times
from config import cfig
from args_parser import parse_args
import os
from config import cfig

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train_nn_baseline():
    shape = (None, 1608)
    if cfig.dtype == 'pca':
        shape = (None, 100)
    prefix = '/baseline/nn'
    runs = cfig.get_n_runs()
    train_n_times(runs, lambda _: MLP([100, 32], shape), load_weights=False, prefix=prefix)
    train_n_times(runs, lambda _: LSTM([100, 500], shape), load_weights=False, prefix=prefix)
    train_n_times(runs, lambda _: CNNModel(0, shape), load_weights=False, prefix=prefix)
    train_n_times(runs, lambda _: CNNModel(1, shape), load_weights=False, prefix=prefix)


if __name__ == "__main__":
    parse_args()
    train_nn_baseline()
    