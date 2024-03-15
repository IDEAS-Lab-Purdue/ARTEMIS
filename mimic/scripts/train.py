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
from mlp import Artemis

# train
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import copy
from tqdm import trange, tqdm

# get data
# mridu shuffled it with sample(frac=1)

data_rebalanced_embeddings = pd.read_csv("./triage_rebalanced_pain_embeddings.csv", sep=",").sample(frac=1)

# convert pain to floats

embeddings_list = []
for i in range(len(data_rebalanced_embeddings)):
  embeddings_list.append([float(i.strip()) for i in data_rebalanced_embeddings.pain_embedding[i][1:-1].split(",")])
data_rebalanced_embeddings['pain_embedding'] = embeddings_list

# condense all attributes into 1 vector

# X = np.zeros((len(data_rebalanced_embeddings), 1541)) # default is float64 so should be good
input_X_text = []
input_X_vitals = []
for i in range(len(data_rebalanced_embeddings)):
  
  text_embedding = np.asarray(data_rebalanced_embeddings.pain_embedding[i])
  vital_attribs = np.asarray([data_rebalanced_embeddings.temperature[i], data_rebalanced_embeddings.heartrate[i],
                              data_rebalanced_embeddings.sbp[i], data_rebalanced_embeddings.dbp[i],
                              data_rebalanced_embeddings.o2sat[i]])
  input_X_text.append(text_embedding)
  input_X_vitals.append(vital_attribs)
  
  
  
  # X[i][0:1536] = array(data_rebalanced_embeddings.pain_embedding[i])
  # X[i][1536] = data_rebalanced_embeddings.temperature[i]
  # X[i][1537] = data_rebalanced_embeddings.heartrate[i]
  # X[i][1538] = data_rebalanced_embeddings.sbp[i]
  # X[i][1539] = data_rebalanced_embeddings.dbp[i]
  # X[i][1540] = data_rebalanced_embeddings.o2sat[i]

input_X_text = torch.from_numpy(np.asarray(input_X_text).astype(np.float32))
print(input_X_text.shape)
input_X_vitals = torch.from_numpy(np.asarray(input_X_vitals).astype(np.float32))
print(input_X_vitals.shape)
y = torch.from_numpy(np.asarray(data_rebalanced_embeddings['acuity']).reshape(-1, 1).astype(np.float32))

'''
train_X_text = torch.from_numpy(input_X_text[:4116])
train_X_vitals = torch.from_numpy(input_X_vitals[:4116])
train_y = torch.from_numpy(y[:4116])

test_X_text = torch.from_numpy(input_X_text[4116:])
test_X_vitals = torch.from_numpy(input_X_vitals[4116:])
test_y = torch.from_numpy(y[4116:])
'''

print(y.shape)

train_set = TensorDataset(input_X_text, input_X_vitals, y)

test_set = TensorDataset(input_X_text, input_X_vitals, y)
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
# loss function
loss_fn = nn.CrossEntropyLoss()

# get model
model = Artemis()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / 5e6))


'''
# loading a saved model
checkpoint = torch.load('./mlp.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
'''

# split data into train and test
# Split the data into features (X) and target (y)
#y = data_rebalanced_embeddings['acuity']

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
      x_text, x_vitals, y = data
      y_pred = model(x_text, x_vitals)
      outputs = [0,0,0,0,0]
      outputs[int(y)-1] = 1.0
      outputs = torch.from_numpy(np.asarray(outputs).astype(np.float32).reshape(1,5))

      train_loss = loss_fn(y_pred,outputs)
      # backward pass
      optimizer.zero_grad()
      train_loss.backward()
      # update weights
      optimizer.step()
      scheduler.step()
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
      x_text, x_vitals, y = data
      y_pred = model(x_text, x_vitals)
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
