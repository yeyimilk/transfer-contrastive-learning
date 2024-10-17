from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from data_loader import load_data
import numpy as np
from sklearn.model_selection import KFold
from utils import save_to_file
from config import cfig
from config import run_args


def train_with_ramens(get_model):
    n_splits = cfig.n_cross
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    x, y, _ = load_data(cfig.dtype)
    
    dic = {}
    accuracy = []
    for i, (train_index, test_index) in enumerate(kf.split(x)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = get_model()
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        pred_y = model.predict(x_test)
        prob = model.predict_proba(x_test)
        dic[f"{i}"] = {
            "accuracy": score,
            "pred_y": pred_y,
            "true_y": y_test,
            "prob": prob
        }
        accuracy.append(score)
    mean = np.mean(accuracy)    
    dic["mean"] = mean
    dic['accuracy'] = mean
    
    return dic
        

def train_mls():
    
    models = [    
        lambda: LogisticRegression(), 
        lambda: SVC(probability=True), 
        lambda: RandomForestClassifier(), 
        lambda: DecisionTreeClassifier(), 
        lambda: KNeighborsClassifier(n_neighbors=3)
    ]
    if cfig.dtype != 'pca':
        models.append(lambda: MultinomialNB())
    
    for get_model in models:
        results = []
        for _ in range(cfig.get_n_runs()):
            result = train_with_ramens(get_model)
            results.append(result)
        name = get_model().__class__.__name__   
        save_to_file(results, f"/baseline/ml_{name}", "json")
    

if __name__ == "__main__":
    run_args()
    train_mls()
