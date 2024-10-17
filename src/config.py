from datetime import datetime
import argparse
class Config:
    def __init__(self) -> None:
        self.pre_epochs = 100
        self.fine_epochs = 100
        self.is_debug = False
        self.n_cross = 5
        self.n_runs = 1
        self.set_augmentations(300, 0.1, 0.1, 100)
        self.t = 0.007
        self.l = [12]
        self.width = [128, 128]
        self.dtype = 'origin'
        self.er = 0
        self.now = datetime.now().strftime('%m%d_%H%M%S')
        
    def get_tag(self):
        return self.tag
        
    def get_pre_train_epochs(self):
        return self.pre_epochs

    def set_pre_train_epochs(self, pre_epochs):
        self.pre_epochs = pre_epochs
        return self
    
    def get_fine_train_epochs(self):
        return self.fine_epochs
    
    def set_fine_train_epochs(self, fine_epochs):
        self.fine_epochs = fine_epochs
        return self
    
    def set_n_cross_v(self, n_cross):
        self.n_cross = n_cross
        return self
    
    def get_n_cross_v(self):
        return self.n_cross
    
    def get_debug(self):
        return self.is_debug
    
    def set_debug(self, is_debug):
        self.is_debug = is_debug
        if is_debug:
            self.pre_epochs = 1
            self.fine_epochs = 1
            self.n_runs = 1
        print(f"debug: {self.is_debug}, pre_epochs: {self.pre_epochs}, \
              fine_epochs: {self.fine_epochs}, dtype: {self.dtype}")
        return self
    
    def set_n_runs(self, n):
        self.n_runs = n
    
    def get_n_runs(self):
        return self.n_runs
    
    def set_augmentations(self, ts, sc, gn, r):
        self.ts = ts
        self.sc = sc
        self.gn = gn
        self.r = r
        return self
    
    def get_augmentations(self):
        return (self.ts, self.sc, self.gn, self.r)
    
    def set_temperature(self, t):
        self.t = t
        return self
    
    def set_layers(self, l):
        ls = l.split(',')
        self.l = [int(i) for i in ls]
        return self

    def get_layers(self):
        return list(self.l)
    
    def set_dtype(self, dtype):
        self.dtype = dtype
        if dtype == 'pca':
            self.set_augmentations(30, 0.1, 0.1, 10)
        return self
    
    def simclr_shape(self):
        shape = (1608, )
        if self.dtype == 'pca':
            shape = (100, )
        return shape
    
    def set_er(self, er):
        self.er = er
        return self

cfig = Config()

def run_args():
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
    parser.add_argument('--tag', type=str, default='', help='tag')
    parser.add_argument('--er', type=int, default=0, help='early stopping')
    
    cfig.set_pre_train_epochs(parser.parse_args().pre)
    cfig.set_fine_train_epochs(parser.parse_args().fine)
    cfig.set_n_runs(parser.parse_args().runs)
    cfig.set_debug(parser.parse_args().debug)
    cfig.set_augmentations(parser.parse_args().ts, parser.parse_args().sc, parser.parse_args().gn, parser.parse_args().r)
    cfig.set_temperature(parser.parse_args().t)
    cfig.set_layers(parser.parse_args().l)
    cfig.set_n_cross_v(parser.parse_args().n_cross)
    cfig.set_dtype(parser.parse_args().dtype)
    cfig.set_er(parser.parse_args().er)
    cfig.tag = parser.parse_args().tag
    cfig.args = parser.parse_args()