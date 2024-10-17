from data_loader import get_cell_data_with_keys
from config import cfig
from utils import make_sure_folder_exists
import pathlib
import numpy as np

def get_name_from_keys_kinds(keys, kinds):
    name = ""
    if keys and kinds:
        name = f"{'_'.join(keys)}_{'_'.join(kinds)}"
        name = name.replace("-", "").replace("(", "").replace(")", "")
    return name

def get_weights_path(cell_name, model_name, prefix=""):
    path_name = f"_{model_name}_{cell_name}"
    abs_path = pathlib.Path(__file__).parent.parent.resolve()
    full_path = f"{abs_path}/data/weights_{cfig.get_pre_train_epochs()}_{cfig.dtype}/{prefix}"
    make_sure_folder_exists(full_path)
    return f"{full_path}{path_name}.h5"

def pre_train_model_with_cells(keys, kinds, model, prefix="", mname=None):
    cell_name = get_name_from_keys_kinds(keys, kinds)
    x, _, y = get_cell_data_with_keys(keys, kinds, dtype=cfig.dtype)
    history = model.fit(x, y,
              epochs=cfig.get_pre_train_epochs(),
              batch_size=1024)
        
    model_name = model.name if mname is None else mname
    path_name = get_weights_path(cell_name, model_name, prefix)
    print(f"save weights to {path_name}")
    model.encoder.save_weights(path_name)
    np.save(f"{path_name}_history", history.history)