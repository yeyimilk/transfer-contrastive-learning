

from data_loader import load_data
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from config import cfig
from utils import save_to_file
from models import Simclr
import tensorflow as tf
from tqdm import tqdm
from keras.callbacks import EarlyStopping

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


def get_er_callbacks():
    early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=20,
                mode='min',
                restore_best_weights=True
        )
    return [early_stopping]
            
def train_encoder_with_simclr(model, x, y):
    aug = cfig.get_augmentations()
    ls = cfig.get_layers()
    ls.append(3)
    
    simclr = Simclr(model.encoder, aug, t=cfig.t, l=ls, width=cfig.width, shape=cfig.simclr_shape())
    c_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    p_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    simclr.compile(c_optimizer, p_optimizer)
    simclr.fit(x, y, 
               epochs=cfig.get_pre_train_epochs(), 
               batch_size=1024)

def train_with_ramens(get_model, load_weights, with_clr=False):    
    x, y, _ = load_data(cfig.dtype)
    
    n_splits = cfig.get_n_cross_v()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    
    dic = {}
    accuracy = []
    
    for i, (train_index, test_index) in tqdm(enumerate(kf.split(x))):
        
        x_train, x_val, y_train, y_val = train_test_split(x[train_index], y[train_index], test_size=1/n_splits, random_state=42, shuffle=True)
        x_test, y_test = x[test_index], y[test_index]
        
        model = get_model(load_weights)
        
        if with_clr:
            train_encoder_with_simclr(model, x_train, y_train)

        callbacks = []
        if cfig.er:
            callbacks = get_er_callbacks()
        else:
            callbacks = [LossHistory()]
            
        history = model.fit(x_train, y_train, 
                  epochs=cfig.get_fine_train_epochs(), 
                  batch_size=x_train.shape[0],
                  validation_data=(x_val, y_val),
                  validation_batch_size=x_val.shape[0],
                  callbacks=callbacks)
        
        score = model.score(x_test, y_test)
        pred_y = model.predict(x_test)
        accuracy.append(score)
        dic[f"{i}"] = {
                "accuracy": score,
                "pred_y": pred_y,
                "true_y": y_test,
                'history': history.history
            }
    
    dic["mean"] = np.mean(accuracy)
    return dic

def train_n_times(n, get_model, load_weights, prefix='', with_clr=False):
    results = []
    for _ in tqdm(range(n), desc=f"Training {n} times in progress...{load_weights}, {prefix}, {with_clr}"):
        result = train_with_ramens(get_model, load_weights, with_clr)
        results.append(result)
        
        if cfig.get_debug():
            break
        
    model =  get_model(load_weights)
    name = model.name
    if prefix:
        postfix = load_weights if load_weights else ""
        name = f"{cfig.dtype}/n_{cfig.n_runs}_pe_{cfig.pre_epochs}_ep_{cfig.fine_epochs}_er_{cfig.er}_tag_{cfig.tag}_{cfig.args.val}/{prefix}_{name}_{postfix}"
    save_to_file(results, name, "json")
    return results