# TCLP
Transfer Contrastive Learning for Raman Spectroscopy Skin Cancer Tissue Classification


# Dataset

### Source data
This dataset was originally collected by Erzina et al.(2020). I have added the dataset to this project while you can also download it from Kaggle, [Cells Raman Spectra](https://www.kaggle.com/datasets/andriitrelin/cells-raman-spectra)


### Target data
Currently, a random dataset was generated for running the baseline code, a simulated dataset generated by GNN which also follows the original distribution will be released after paper has been accepted.
# Run

### ML baseline
```
cd src
python ml_baseline
```
This command runs the 6 traditional machine learning models, LogisticRegression, SVC, RandomForestClassifier, DecisionTreeClassifier, KNeighborsClassifier

### NN baseline
```
cd src
python nn_baseline
```

This commonds runs the 4 neural network models, including 1 MLP, 1 LSTM, and 2 CNNs

### TCLP
This part will be released after paper has been accepted 