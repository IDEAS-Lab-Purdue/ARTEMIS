# Data Processing
import pandas as pd
import numpy as np
from numpy import array

# Modelling
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, classification_report, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# save model to file
import pickle

# model
from mlp_smote import Artemis

# train
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import copy
from tqdm import trange, tqdm

# get data

# datasets
# triage_smote: oversampled 5 to 2000 and undersampled the rest to 2000 (20% acc)
# triage_smote2: with resprate (20% acc)
# triage_smote3: oversample all to 216424 (25% ish)
# triage_smote4: all are 77651 (20% acc)
# triage_smote5: leave acuity=5 to 1029 and rest become 96807 (20% acc)
# triage_smote6: make all 4 216424 and drop acuity=5 (25% acc)

data = pd.read_csv("./triage_smote6.csv", sep=",").sample(frac=1)

# condense all attributes into 1 vector

input_X_vitals = []
for i in range(len(data)):
  vital_attribs = np.asarray([data.temperature[i], data.heartrate[i], data.resprate[i],
                              data.sbp[i], data.dbp[i],
                              data.o2sat[i]])
  input_X_vitals.append(vital_attribs)

input_X_vitals = torch.from_numpy(np.asarray(input_X_vitals).astype(np.float32))
#print(input_X_vitals.shape)
y = torch.from_numpy(np.asarray(data['acuity']).reshape(-1, 1).astype(np.float32))
#print("Y = ", y.shape)
train_samples = int(0.8*len(input_X_vitals))
train_set = TensorDataset(input_X_vitals[:train_samples], y[:train_samples])
test_set = TensorDataset(input_X_vitals[train_samples:], y[train_samples:])
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
# loss function
loss_fn = nn.MSELoss()

# get model
model = Artemis()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / 5e6))


'''
# loading a saved model
checkpoint = torch.load('./mlp.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
'''

# split data into train and test
# Split the data into features (X) and target (y)
#y = data['acuity']

# Split the data into training and test sets
#X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

#X_train = torch.tensor(X_tr, dtype=torch.float64)
#y_train = torch.tensor(y_tr.to_numpy().reshape(-1,1), dtype=torch.long)
#X_test = torch.tensor(X_te, dtype=torch.float64)
#y_test = torch.tensor(y_te.to_numpy().reshape(-1,1), dtype=torch.long)

# convert to one hot vector
#ohe = OneHotEncoder(handle_unknown='ignore', sparse=False).fit(y_train)
#y_train = torch.from_numpy(ohe.transform(y_train))
#print(ohe.categories_)
#ohe = OneHotEncoder(handle_unknown='ignore', sparse=False).fit(y_test)
#y_test = torch.from_numpy(ohe.transform(y_test))

# train

epochs = 200
#train_total = len(X_train)
#test_total = len(X_test)
best_test_acc = - np.inf
best_train_acc = - np.inf
best_weights = None

for epoch in range(epochs):
    test_acc = 0.0
    train_acc = 0.0
    test_loss = 0.0
    train_loss = 0.0
    print(f"Epoch {epoch}:")
    batch_counter = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for data in progress_bar:
      # forward pass
      x_vitals, y = data
      #x_vitals = x_vitals.reshape(-1, 1)
      
      y_pred = model(x_vitals)
      outputs = [0,0,0,0,0]
      outputs[int(y)-1] = 1.0
      outputs = torch.from_numpy(np.asarray(outputs).astype(np.float32).reshape(1,5))

      train_loss = loss_fn(y_pred,outputs)
      # backward pass
      optimizer.zero_grad()
      train_loss.backward()
      # update weights
      optimizer.step()
      #scheduler.step()
      # compute and store metrics
      #print("Y PRED = ", y_pred)
      #print("Model Pred = ", torch.argmax(y_pred))
      #print("Outputs = ", torch.argmax(outputs))
      #print("Y = ", y)
      train_acc += (torch.argmax(y_pred) == torch.argmax(outputs)).float()
      #print("TRAIN ACC = ", train_acc)
      batch_counter += 1
      progress_bar.set_postfix({"loss": train_loss.item(), "avg_accuracy": train_acc / batch_counter, "lr": optimizer.param_groups[0]["lr"]})

    train_acc /= batch_counter
    train_loss = float(train_loss)
    train_acc = float(train_acc)
    print(f"Train Cross-Entropy={train_loss}, Train Accuracy={train_acc}")
    progress_bar = tqdm(test_loader, desc=f"Epoch {epoch}")
    batch_counter = 0
    for data in progress_bar:
      x_vitals, y = data
      y_pred = model(x_vitals)
      outputs = [0,0,0,0,0]
      outputs[int(y)-1] = 1.0
      outputs = torch.from_numpy(np.asarray(outputs).astype(np.float32).reshape(1, 5))
      test_loss = loss_fn(y_pred, outputs)
      test_acc += (torch.argmax(y_pred) == torch.argmax(outputs)).float()
      batch_counter += 1
      progress_bar.set_postfix({"loss": test_loss.item(), "avg_accuracy": test_acc / batch_counter, "lr": optimizer.param_groups[0]["lr"]})
    test_acc /= batch_counter
    test_acc = float(test_acc)
    test_loss = float(test_loss)
    if test_acc > best_test_acc and train_acc > best_train_acc:
      best_train_acc = train_acc
      best_test_acc = test_acc
      # save the model
      torch.save({
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'optimizer_state_dict': copy.deepcopy(optimizer.state_dict())}, './mlp.pt')
    print(f"Test Cross-Entropy={test_loss}, Test Accuracy={test_acc}")
