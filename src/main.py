import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import random
import os
from sklearn.metrics import classification_report

# Параметры спектрограммы
IMG_HEIGHT = 128
IMG_WIDTH = 128
CHANNEL_CENTER_BIN = {1: 12, 2: 26, 3: 40, 4: 54}



class WifiDetector(nn.Module):
    def __init__(self):
        super(WifiDetector, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*32*32, 128),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 16*32*32),
            nn.Unflatten(1, (16, 32, 32)),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        logits = self.classifier(features)
        return reconstruction, logits


