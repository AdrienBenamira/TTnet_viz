import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, EarlyStopping

from pages.modules.helper import read_info, read_csv, DBEncoder, find_splits, sample_generator

from ttnet.TT_preprocessing import preprocessing_QUANT_BN_STEP
from ttnet.load_datasets import get_data_loader
from ttnet.modules_TT import Binarize01Act, Block_TT_1D, Classifier_continu
from ttnet.modules_general import SeqBinModelHelper, BinLinearPos, g_weight_binarizer, BinLinearPosv2, setattr_inplace, \
    BatchNormStatsCallbak, g_use_scalar_scale_last_layer
from ttnet.utils import ModelHelper, Flatten

lin = BinLinearPos
lin2 = BinLinearPosv2
wb = g_weight_binarizer
act = Binarize01Act



class ClassifierModule(nn.Module):
    def __init__(
            self, k=5  # kernel size < 9
            , s=5  # stride
            , p=2  # padding
            , repeat=3, filter_size=5, embed_size=10
    ):
        super(ClassifierModule, self).__init__()
        # self.BNS = []
        # for r in range(repeat):
        # self.BNS.append(nn.BatchNorm1d(1))
        self.BN0 = nn.BatchNorm1d(1)
        self.BN1 = nn.BatchNorm1d(1)
        self.BN2 = nn.BatchNorm1d(1)
        self.repeat = repeat
        self.conv1 = nn.Conv1d(1, filter_size, kernel_size=k,
                                   stride=s, padding=p, groups=1, bias=True)
        self.nonlin = act()
        with torch.no_grad():
            x_r = torch.zeros((1, 100))
            x_r2 = self.block(self.preprocess(x_r))
            x_r2 = x_r2.reshape(x_r2.shape[0], -1)
        print("Number of rules, ", x_r2.shape[-1], " size rules ", k)
        self.dense1 = nn.Linear(x_r2.shape[-1], embed_size)  # (260,10)
        self.dense2 = nn.Linear(embed_size, embed_size)
        self.output = nn.Linear(embed_size, 2)

    def preprocess(self, X, **kwargs):
        X_d, X_c = X[:, :-6].clone().unsqueeze(1).float(), X[:, 94:].clone()
        X_c0 = self.nonlin(self.BN0(X_c.unsqueeze(1).float()))
        X_c1 = self.nonlin(self.BN1(X_c.unsqueeze(1).float()))
        X_c2 = self.nonlin(self.BN2(X_c.unsqueeze(1).float()))
        if self.repeat == 1:
            X = torch.cat((X_d, X_c0), axis=2)
        if self.repeat == 2:
            X = torch.cat((X_d, X_c0, X_c1), axis=2)
        if self.repeat == 3:
            X = torch.cat((X_d, X_c0, X_c1, X_c2), axis=2)
        else:
            raise "PB"
        return X

    def block(self, X, **kwargs):
        X = self.nonlin(self.conv1(X.float()))
        return X

    def forward(self, X, **kwargs):
        X = self.block(self.preprocess(X))
        X = X.reshape(X.shape[0], -1)
        X = self.dense1(X)
        X = self.dense2(X)
        X = F.softmax(self.output(X), dim=-1)
        return X

device = "cuda"

lr_scheduler = LRScheduler(policy=ReduceLROnPlateau, monitor='valid_loss', mode='min', patience=3, factor=0.1,
                           verbose=True)
early_stopping = EarlyStopping(monitor='valid_loss', patience=5, threshold=0.0001, threshold_mode='rel',
                               lower_is_better=True)

net = NeuralNetClassifier(ClassifierModule, device=device, max_epochs=50, callbacks=[lr_scheduler, early_stopping])


def train(k, f, gen_data, labels):
    grid_params = {
        'net__lr': [0.03],  # [0.0005, 0.001, 0.005, 0.01, 0.02, 0.03]
        'net__optimizer': [optim.Adam],  # [optim.AdamW, optim.Adam]
        'net__module__k': [k],  # [i for i in range(3,10)],
        'net__module__s': [3],
        'net__module__p': [2],
        'net__module__repeat': [3],
        'net__module__filter_size': [f],  # [i for i in range(2,21)],
        'net__module__embed_size': [10],
        'net__batch_size': [64]
    }
    steps = [('net', net)]
    pipeline = Pipeline(steps)

    grid_net = GridSearchCV(pipeline, grid_params, cv=5, refit=True, verbose=1)
    results = grid_net.fit(gen_data, labels)
    return grid_net, results

