from models import StraightNN, ResNN, RNN_, MLP_, CNN_O, Simclr
from trainer_utils import train_n_times
from pre_train import pre_train_model_with_cells, get_name_from_keys_kinds, get_weights_path
import tensorflow as tf
from config import cfig
from config import run_args

Combinations = {
            'keys': ['MEL-S', 'A-S', 'G', 'A', 'ZAM', 'G-S', 'DMEM-S', 'HF', 'HF-S', 'ZAM-S', 'MEL', 'DMEM'],
            'kinds': ['COOH', 'NH2', '(COOH)2']
    }

def train_pretrain_models(prefix, get_model):
    keys = Combinations['keys']
    kinds = Combinations['kinds']
    cell_name = get_name_from_keys_kinds(keys, kinds)
    setup_and_train_model_with_weights(get_model, cfig.n_runs, prefix, cell_name)
        

def setup_and_train_model_with_weights(get_model, times, prefix, load_weights):
    def load_model(_):
        model = get_model(False)
        path_name = get_weights_path(load_weights, model.name, prefix)
        model.load_weights_for_encoder(path_name)
        return model
    train_n_times(times, load_model, load_weights=load_weights, prefix=prefix)
 

def evaluate_models(get_model):
    prefix = 'models/base'
    train_n_times(cfig.n_runs, get_model, load_weights=False, prefix=prefix)
       
def evaluate_transfer_pretrain_models(get_model):
    prefix = 'transfer_pretrain/'
    keys = Combinations['keys']
    kinds = Combinations['kinds']
    model = get_model(len(keys))
    pre_train_model_with_cells(keys, kinds, model, prefix=prefix)
        
        
    train_pretrain_models(prefix, get_model)

def evaluate_simclr_pretrain_models(get_model):
    prefix = 'simclr_pretrain/'
    
    def setup_and_pretrain_simclr(get_model, keys, kinds, prefix):
        model = get_model(False)
        aug = cfig.get_augmentations()
        ls = cfig.get_layers()
        head = len(keys) if len(keys) > 3 else 3
        ls.append(head)
        simclr = Simclr(model.encoder, aug, t=cfig.t, l=ls, width=cfig.width, shape=cfig.simclr_shape())
        c_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        p_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        simclr.compile(c_optimizer, p_optimizer)
        pre_train_model_with_cells(keys, kinds, simclr, prefix=prefix, mname=model.name)
    
    for com in Combinations:
        keys = com['keys']
        kinds = com['kinds']
        setup_and_pretrain_simclr(get_model, keys, kinds, prefix)
        if cfig.get_debug():
            break
    train_pretrain_models(prefix, get_model)

def evaluate_simclr_train_models(get_model):
    prefix = 'simclr_train/'
    train_n_times(cfig.n_runs, get_model, 
                  load_weights=False, prefix=prefix, with_clr=True)


def evalute_model(get_model):
    evaluate_models(get_model)
    evaluate_transfer_pretrain_models(get_model)
    evaluate_simclr_train_models(get_model)
    evaluate_simclr_pretrain_models(get_model)

def evalutes():
    shape = (1608, )
    if cfig.dtype == 'pca':
        shape = (100, )
        
    evalute_model(lambda _: StraightNN([512], [7], [1], 500, shape, out = _ if type(_) == int else 3))
    evalute_model(lambda _: StraightNN([500], [5], [1], 512, shape, out = _ if type(_) == int else 3))
    evalute_model(lambda _: ResNN([64, 64, 64], [3, 5, 7], [1, 2, 4], 500, shape, out = _ if type(_) == int else 3))
    evalute_model(lambda _: MLP_([100, 32], shape, out = _ if type(_) == int else 3))
    evalute_model(lambda _: CNN_O(shape, out = _ if type(_) == int else 3))
    
    evalute_model(lambda _: RNN_([100], shape, [True], False, 'SimpleRNN', out = _ if type(_) == int else 3))
    evalute_model(lambda _: RNN_([100], shape, [True], True, 'GRU', out = _ if type(_) == int else 3))
    evalute_model(lambda _: RNN_([100], shape, [True], False, 'LSTM', out = _ if type(_) == int else 3))


if __name__ == "__main__":
    parse_args()
    evalutes()