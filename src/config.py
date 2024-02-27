from datetime import datetime
class Config:
    def __init__(self) -> None:
        self.pre_epochs = 100
        self.fine_epochs = 100
        self.is_debug = False
        self.n_cross = 5
        self.val_size = 0.1
        self.n_runs = 20
        self.set_augmentations(300, 0.1, 0.1, 100)
        self.t = 0.007
        self.l = [12]
        self.width = [128, 128]
        self.dtype = 'origin'
        self.mv = 5
        self.wave = 0
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
    
    def set_val_size(self, val_size):
        self.val_size = val_size
        return self
    
    def get_val_size(self):
        return self.val_size
    
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
    
    def get_tag(self):
        self.tag = f"{self.dtype}/t_{self.now}"
        return self.tag
    
    def simclr_shape(self):
        shape = (1608, )
        if self.dtype == 'pca':
            shape = (100, )
        return shape
    
    def set_mv(self, mv):
        self.mv = mv
        return self
    
    def set_wave(self, wave):
        self.wave = wave
        return self

cfig = Config()