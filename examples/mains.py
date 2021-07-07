import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from torchfm.dataset.avazu import AvazuDataset
from torchfm.dataset.criteo import CriteoDataset
from torchfm.dataset.movielens import MovieLens1MDataset, MovieLens20MDataset
from torchfm.model.afi import AutomaticFeatureInteractionModel
from torchfm.model.afm import AttentionalFactorizationMachineModel
from torchfm.model.dcn import DeepCrossNetworkModel
from torchfm.model.dfm import DeepFactorizationMachineModel
from torchfm.model.ffm import FieldAwareFactorizationMachineModel
from torchfm.model.fm import FactorizationMachineModel
from torchfm.model.fnfm import FieldAwareNeuralFactorizationMachineModel
from torchfm.model.fnn import FactorizationSupportedNeuralNetworkModel
from torchfm.model.hofm import HighOrderFactorizationMachineModel
from torchfm.model.lr import LogisticRegressionModel
from torchfm.model.ncf import NeuralCollaborativeFiltering
from torchfm.model.nfm import NeuralFactorizationMachineModel
from torchfm.model.pnn import ProductNeuralNetworkModel
from torchfm.model.wd import WideAndDeepModel
from torchfm.model.xdfm import ExtremeDeepFactorizationMachineModel
from torchfm.model.afn import AdaptiveFactorizationNetwork
from torchfm.layer import FeaturesEmbedding
import pandas as pd
import numpy as np
from examples.function import get_dataset, get_model, EarlyStopper


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='movielens1M')
    parser.add_argument('--dataset_path', default='ml-1m/ratings.dat')
    parser.add_argument('--model_name', default='fm')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_dir', default='chkpt')
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset = get_dataset(args.dataset_name, args.dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8)

    field_dims = dataset.field_dims
    model = get_model(args.model_name, dataset)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay
                                 )
    early_stopper = EarlyStopper(num_trials=2, save_path=f'{args.save_dir}/{args.model_name}.pt')
    log_interval = 100
    loss_list = []
    for epoch_i in range(args.epoch):
        model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(train_data_loader, smoothing=0, mininterval=1.0)
        for i, (fields, target) in enumerate(tk0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            loss = criterion(y, target.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loss_list.append(loss.item())
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

    model2 = get_model(args.model_name, dataset)
    d = train_data_loader.dataset.dataset.items
    ds = torch.from_numpy(d)
    e = model.embedding(ds)
    s = e.detach().numpy()
    p = s.reshape(s.shape[0], -1)
    # print(f'test auc: {auc}')
    import matplotlib.pyplot as plt
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()

    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(test_data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    auc = roc_auc_score(targets, predicts)
    print(f'test auc: {auc}')