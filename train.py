import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tool import mat2graph
from models import NonLocalGCNLayer
from sklearn.model_selection import StratifiedKFold
import random
import scipy.sparse as sp
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from scipy import sparse
import argparse


def get_lambda_max(adj):
    rst = []
    n = adj.shape[0]
    in_degrees = np.sum(adj, 0).astype(float)
    adj = sparse.csr_matrix(adj).astype(float)
    norm = sparse.diags(in_degrees.clip(1) ** -0.5, dtype=float)
    laplacian = sparse.eye(n) - norm * adj * norm
    rst.append(sparse.linalg.eigs(laplacian, 1, which='LM', return_eigenvectors=False)[0].real)
    return rst


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def feature_selection(feat, labels, train_ind, dim=2000):
    estimator = RidgeClassifier(500)
    selector = RFE(estimator, n_features_to_select=dim, step=100, verbose=0)
    featureX = feat[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(feat)
    return x_data


def train_epoch(model, optimizer, loss_func, x, y, index):
    model.train()
    y_pred = model(x)
    loss = loss_func(y_pred[index], y[index])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def test_epoch(model, loss_func, x, y, index):
    with torch.no_grad():
        model.eval()
        y_pred = model(x)
        loss = loss_func(y_pred[index], y[index])
        acc_num = (torch.argmax(y_pred[index], 1) == y[index]).sum().item()
        return acc_num / len(index), loss.item()


def GCN_train(x, y, train_idx, test_idx, parser):

    args = parser.parse_args()
    lr = args.lr
    weight_decay = args.weight_decay
    dropout = args.dropout
    hidden_feat = args.hidden_feat
    epochs = args.epochs
    num_head = args.num_head
    factor = args.factor
    patience = args.patience
    seed = args.seed

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    adj = np.ones((x.shape[0], x.shape[0])) - np.eye(x.shape[0])

    device = torch.device('cuda:0')

    G = mat2graph(adj)
    x = feature_selection(x, y, train_idx)
    x = torch.from_numpy(preprocess_features(x)).float()
    y = torch.tensor(y).long()


    class GCN_model(nn.Module):
        def __init__(self, G, init_feat, hidden_feat, n_class,
                     num_head=1, feat_drop=0.0):
            super(GCN_model, self).__init__()

            self.G = G.to(device)
            self.gcn_layer1 = NonLocalGCNLayer(in_feats=init_feat, out_feats=hidden_feat, num_heads=num_head,
                                               feat_drop=feat_drop, activation=nn.PReLU(), agg_mode='flatten')

            self.classify_layer = nn.Linear(num_head * hidden_feat, n_class)

        def forward(self, x):
            feat = x
            feat = self.gcn_layer1(self.G, feat)
            feat = self.classify_layer(feat)
            return feat

    init_feat = x.shape[1]
    n_class = 2
    model = GCN_model(G=G, init_feat=init_feat, hidden_feat=hidden_feat, n_class=n_class,
                      feat_drop=dropout, num_head=num_head)
    optimizer = optim.Adam([
                            {'params': model.gcn_layer1.parameters(), 'weight_decay': weight_decay},
                            {'params': model.classify_layer.parameters()},
                            ], lr=lr)
    loss_func = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience)

    model.to(device)
    x = x.to(device)
    y = y.to(device)

    train_losses = []
    accuracy = []

    for epoch in range(1, epochs+1):
        train_loss = train_epoch(model, optimizer, loss_func, x, y, train_idx)
        train_losses.append(train_loss)
        acc, _ = test_epoch(model, loss_func, x, y, test_idx)
        accuracy.append(acc)
        scheduler.step(accuracy[-1])

    return accuracy[-1]


def main():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    x, y = pickle.load(open('./data/871_feat.pkl', 'rb')), pickle.load(open('./data/labels.pkl', 'rb'))
    y = y - 1

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--lr', type=float, default=0.065)
    parser.add_argument('--weight_decay', type=float, default=2.5e-4)
    parser.add_argument('--hidden_feat', type=float, help='hidden representation dim', default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num_head', type=int, default=1)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--factor', type=float, help='lr decline rate', default=0.85)
    parser.add_argument('--dropout', type=float, default=0.25)

    accs = []
    for train_idx, test_idx in skf.split(x, y):
        acc = GCN_train(x, y, train_idx, test_idx, parser=parser)
        accs.append(acc)
    print(accs)
    print(np.mean(accs), np.std(accs))


if __name__ == "__main__":
    main()


