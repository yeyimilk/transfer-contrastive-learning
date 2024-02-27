

from data_loader import load_data
import numpy as np
from sklearn.model_selection import KFold
from config import cfig
from utils import save_to_file
from tqdm import tqdm


def train_with_ramens(get_model, load_weights, with_clr=False):
    n_splits = cfig.get_n_cross_v()
    
    x, y, _ = load_data(cfig.dtype)
    
    
    if n_splits == 0:
        n_splits = x.shape[0] # leave one out
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    dic = {}
    accuracy = []
    for i, (train_index, test_index) in tqdm(enumerate(kf.split(x))):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        x_val, y_val = x_test, y_test
        
        model = get_model(load_weights)
        
        model.fit(x_train, y_train, 
              epochs=cfig.get_fine_train_epochs(), 
              batch_size=x_train.shape[0],
              validation_data=(x_val, y_val),
              validation_batch_size=x_val.shape[0],
              validation_freq=100,
              verbose=True)
        
        score = model.score(x_test, y_test)
        pred_y = model.predict(x_test)
        dic[f"{i}"] = {
            "accuracy": score,
            "pred_y": pred_y,
            "true_y": y_test
        }
        accuracy.append(score)
        
        if cfig.get_debug():
            break
        
    mean = np.mean(accuracy)    
    dic["mean"] = mean
    
    print(f"accuracy: {accuracy}, mean: {mean}")
    
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
        name = f"{cfig.n_runs}_{cfig.pre_epochs}_{cfig.fine_epochs}/{prefix}_{name}_{postfix}"
    save_to_file(results, name, "json")
    return results