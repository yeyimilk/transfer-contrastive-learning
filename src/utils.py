import os
import pathlib
from datetime import datetime
import json
import numpy as np
from config import cfig

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def make_sure_folder_exists(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    
def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")

def make_path(name):
    name = name_with_datetime(name)
    abs_path = pathlib.Path(__file__).parent.resolve()
    
    sub_path = os.path.dirname(name)
    file_dir = f'{abs_path}/results/{cfig.get_tag()}/{sub_path}'
    make_sure_folder_exists(file_dir)
    return abs_path, name, file_dir

def save_to_file(content, name, type='txt'):
    abs_path, name, file_dir = make_path(name)
    
    if type == 'json':
        content = json.dumps(content, cls=NpEncoder)
    file_path = f'{abs_path}/results/{cfig.get_tag()}/{name}.{type}'
    with open(file_path, 'w') as f:
        f.write(content)
    
    return file_path