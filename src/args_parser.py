import argparse
from config import cfig

def parse_args():
    parser = argparse.ArgumentParser(description='Process parameters.')
    parser.add_argument('--pre', type=int, default=50, help='pre train epochs')
    parser.add_argument('--fine', type=int, default=300, help='fine train epochs')
    parser.add_argument('--runs', type=int, default=20, help='number of runs')
    parser.add_argument('--debug', type=bool, default=False, help='debug mode')
    parser.add_argument('--ts', type=int, default=300, help='augmentation time shift')
    parser.add_argument('--sc', type=float, default=0.1, help='augmentation scale')
    parser.add_argument('--gn', type=float, default=0.1, help='augmentation gaussian noise')
    parser.add_argument('--r', type=float, default=100, help='augmentation reverse')
    parser.add_argument('--t', type=float, default=0.007, help='temperature')
    parser.add_argument('--l', type=str, default='12', help='layers')
    parser.add_argument('--n_cross', type=int, default=5, help='n cross valiation, 0 is one out')
    parser.add_argument('--dtype', type=str, default='origin', help='data type, origin or pca or mov_ave')
    parser.add_argument('--mv', type=int, default=20, help='window size')
    parser.add_argument('--wave', type=int, default=0, help='wavelet, 0 nothing, 1 hard extend, 2 smooth extend')
    
    cfig.set_pre_train_epochs(parser.parse_args().pre)
    cfig.set_fine_train_epochs(parser.parse_args().fine)
    cfig.set_n_runs(parser.parse_args().runs)
    cfig.set_debug(parser.parse_args().debug)
    cfig.set_augmentations(parser.parse_args().ts, parser.parse_args().sc, parser.parse_args().gn, parser.parse_args().r)
    cfig.set_temperature(parser.parse_args().t)
    cfig.set_layers(parser.parse_args().l)
    cfig.set_n_cross_v(parser.parse_args().n_cross)
    cfig.set_dtype(parser.parse_args().dtype)
    cfig.set_mv(parser.parse_args().mv)
    cfig.set_wave(parser.parse_args().wave)