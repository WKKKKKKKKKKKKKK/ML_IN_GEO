# data_utils.py

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pywt
import numpy as np
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import torch.nn as nn


# ==============================
# Transform
# ==============================
def fft_transform(X):
    X_fft = torch.fft.rfft(X, dim=1)
    return torch.abs(X_fft)

def stft_transform(X, n_fft=256):

    X_list = []

    for sample in X:
        stft = torch.stft(sample, n_fft=n_fft, return_complex=True)
        X_list.append(torch.abs(stft).flatten())

    return torch.stack(X_list)

def wavelet_transform(X, wavelet='db4'):

    X_list = []

    for sample in X:
        coeffs = pywt.wavedec(sample.numpy(), wavelet, level=4)
        features = np.concatenate(coeffs)
        X_list.append(features)

    return torch.tensor(np.array(X_list), dtype=torch.float32)

def spectrogram_transform(X, n_fft=256, hop_length=128, return_power=True):
    """
    X: Tensor, shape = (N, T)
    n_fft: FFT window size
    hop_length: step size
    return_power: True -> return power spectrogram (magnitude^2)
    """

    
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)

    X_list = []

    for sample in X:

        
        stft = torch.stft(
            sample,
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True
        )

    
        magnitude = torch.abs(stft)

        
        if return_power:
            magnitude = magnitude ** 2

        
        X_list.append(magnitude.flatten())

    return torch.stack(X_list)


# ==============================
# Dataset
# ==============================

class MyDataset(Dataset):
    def __init__(self, data, labels):

        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)

        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.float32)

        self.data = data.float()
        self.labels = labels.float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ==============================
# Loader
# ==============================

def create_dataloaders(
        data,
        labels,
        batch_size=8,
        test_size=0.2,
        random_state=42,
        shuffle_train=True,
        transform=None,
        Reduce_the_size=None
    ):

    #  tensor
    if not torch.is_tensor(data):
        data = torch.tensor(data, dtype=torch.float32)

    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, dtype=torch.float32)

    # -------- split data --------
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )
    if Reduce_the_size is not None: 
        train_data, _, train_labels, _ = train_test_split(train_data, train_labels, train_size=Reduce_the_size,
        random_state=random_state
    )


    # -------- transform --------
    if transform is not None:
        train_data = transform(train_data)
        test_data = transform(test_data)

    # -------- construct Dataset --------
    dataset_train = MyDataset(train_data, train_labels)
    dataset_test  = MyDataset(test_data, test_labels)

    # -------- DataLoader --------
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=shuffle_train
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


# ==============================
# Training loop
# ==============================

# Sample-based detetion
def train_model(model, train_loader, test_loader, epochs=30):

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        total_batches = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()

            outputs = model(xb)
            loss = criterion(outputs, yb)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        avg_loss = running_loss / total_batches

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")

    return evaluate(model, test_loader)

def evaluate(model, test_loader):

    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            outputs = model(xb)

            probs = outputs.numpy()
            preds = (outputs > 0.5).float().numpy()

            y_prob.extend(probs)
            y_pred.extend(preds)
            y_true.extend(yb.numpy())

    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    y_prob = np.array(y_prob).ravel()

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_prob)
    }


#Picking
def train_model_picking(model, train_loader, test_loader, epochs=30):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Threshold
    TH = 0.5

    # Train the model
    model.train()

    for epoch in range(epochs):

        for i, (images, labels) in enumerate(train_loader):

            # Flatten + GPU + requires_grad 
            images = images.view(images.size(0), -1).requires_grad_(True).to(device)
            labels = labels.view(images.size(0), -1).to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(images)

            # Loss
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()

            # Update
            optimizer.step()

            # Threshold predictions
            predicted = np.where(
                outputs.detach().cpu().numpy() > TH,
                1,
                0
            )

        # Print Loss
        print('epoch: {}. Loss: {}'.format(epoch, loss.item()))

    return evaluate_picking(model, test_loader, device=device)


def evaluate_picking(model, test_loader, samp=50, device=None):

    model.eval()

    laball = []
    preall = []
    dat = []
    outputsall = []

    TH = 0.5

    with torch.no_grad():
        for images, labels in test_loader:

            # view + requires_grad + to(device)
            images = images.view(images.size(0), -1).requires_grad_(True).to(device)
            labels = labels.view(images.size(0), -1).to(device)

            # forward
            outputs = model(images)

            # threshold
            predicted = np.where(
                outputs.detach().cpu().numpy() > TH,
                1,
                0
            )

            laball.append(labels.cpu().numpy())
            preall.append(predicted)
            dat.append(images.detach().cpu().numpy())
            outputsall.append(outputs.detach().cpu().numpy())

    # concatenate
    preall = np.concatenate(preall)
    laball = np.concatenate(laball)
    outputsall = np.concatenate(outputsall)
    dat = np.concatenate(dat)

    # =========================
    # Picking
    # =========================
    P_wave_Pred = preall.argmax(axis=-1)
    P_wave_True = laball.argmax(axis=-1)

    pwave = []
    pwavetp=[]
    pwavetn=[]
    pwavefp=[]
    pwavefn=[]
    fals=[]

    for iq in range(len(P_wave_True)):
    
        if (P_wave_Pred[iq]!=0) and (P_wave_True[iq]!=0):
            pwavetp.append(P_wave_True[iq]-P_wave_Pred[iq])
        
        elif (P_wave_Pred[iq]==0) and (P_wave_True[iq]!=0):
            pwavefn.append(iq)
        
        elif (P_wave_Pred[iq]==0) and (P_wave_True[iq]==0):
            pwavetn.append(iq)
    
        elif (P_wave_Pred[iq]!=0) and (P_wave_True[iq]==0):
            pwavefp.append(iq)
            fals.append(iq)



    samp = 50
    diftp = np.array(pwavetp)
    TP = len(diftp)
    TN = len(pwavetn) 
    FP = len(pwavefp) 
    FN = len(pwavefn) + len(np.where(np.abs(diftp)>samp)[0])
    P = TP /(TP+FP)
    R = TP / (TP+FN)
    F1 = 2 * (P*R) / (P+R)
    
    return {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "Precision": P,
        "Recall": R,
        "F1-score": F1,
    }