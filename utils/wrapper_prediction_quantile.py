from sklearn import metrics
import torch
import pandas as pd
import pickle
import glob
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from core.dataset import GHLDataset
from core.model_prediction import TemporalFusionTransformer
# from core.model import TemporalFusionTransformer
from utils.criterion import QuantileLoss, QuantileLoss_Test

bar_format = '{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'

def train_model(path, args, config):
    
    EPOCHS = args.EPOCHS
    device = args.device
    state_file = args.state_file
    best_valid_loss = float('inf')

    dataset = GHLDataset(path, config, train=True)
    dataset_split_n = int(len(dataset) / 10)
    train_data, val_data = random_split(dataset, [dataset_split_n*8, len(dataset)-(dataset_split_n*8)])
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
    valid_loader = DataLoader(val_data, args.batch_size, shuffle=True)
    
    # train_iterator = tqdm(train_loader, total=len(train_loader), desc="train", leave=False)


    model = TemporalFusionTransformer(config).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    # criterion = nn.MSELoss(reduction='mean')
    criterion = QuantileLoss(config).to(device)
    model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0
        val_loss = 0
        with tqdm(train_loader, desc="Epoch {}/{}".format(epoch + 1, EPOCHS), bar_format=bar_format) as tqdm_loader:
            for i, batch in enumerate(tqdm_loader):
                batch = {key: tensor.to(device) if tensor.numel() else None for key, tensor in batch.items()}

                targets = batch['target'][:, config.encoder_length:, :]
                outputs = model(batch)
                outputs = outputs.view(-1, config.example_length-config.encoder_length, config.output_size, len(config.quantiles))
                targets = targets.unsqueeze(3)
                loss = criterion(outputs, targets)
                loss = loss.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * outputs.size(0)
                if (i + 1) == len(train_loader):
                    tqdm_loader.set_postfix(dict(loss=(epoch_loss / len(train_data))))
                else:
                    tqdm_loader.set_postfix(loss=loss.item())
        with torch.no_grad():
            for batch in valid_loader:
                batch = {key: tensor.to(device) if tensor.numel() else None for key, tensor in batch.items()}
            
                targets = batch['target'][:, config.encoder_length:, :]
                outputs = model(batch)
                outputs = outputs.view(-1, config.example_length-config.encoder_length, config.output_size, len(config.quantiles))
                targets = targets.unsqueeze(3)
                loss = criterion(outputs, targets)
                loss = loss.sum()
                val_loss += loss.mean().item()
            
            print("Validation Loss : {}".format(val_loss))

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), state_file)
        
    print("state_file: {}".format(state_file))
    
def test_model(path, args, config, save=True, attention=False):
    
    device = args.device
    state_file = args.state_file
    
    test_files  = sorted([file for file in glob.glob(path + '*.csv')])
    prefix      = "results/result_w_attn" if attention is True else "results/result_wo_attn"
    postfix     = "ex_{}_enc_{}_epochs_{}".format(config.example_length, config.encoder_length, args.EPOCHS)
    result_file = "{}_{}.pkl".format(prefix, postfix)

    model = TemporalFusionTransformer(config).to(device)
    model.load_state_dict(torch.load(state_file))
    print("state_file: {}".format(state_file))
    print("--")
    criterion = QuantileLoss_Test(config).to(device)

    roc_aucs = []
    pr_aucs = []    
    
    for test_file in test_files:
        test_data = GHLDataset(test_file, config, train=False, dataset=args.dataset)
        test_loader = DataLoader(test_data, args.batch_size, shuffle=False)
        print("test_file: {}".format(test_file.split('/')[-1]))
        
        scores = []
        with torch.no_grad():
            model.eval()

            for batch, _ in test_loader:
                batch = {key: tensor.to(device) if tensor.numel() else None for key, tensor in batch.items()}
                targets = batch['target'][:, config.encoder_length:, :]

                predicts = model(batch)
                predicts = predicts.view(-1, config.example_length-config.encoder_length, config.output_size, len(config.quantiles))
                targets = targets.unsqueeze(3)
                score = criterion(predicts, targets).cpu()
                # score = get_mse_loss(targets, predicts).cpu()
                score = score.sum(1)
                scores.extend(score)
        
        labels = test_data.labels
        positive = sum(labels)
        negative = len(labels) - positive
        roc_auc = metrics.roc_auc_score(labels, scores)
        pr_auc = metrics.average_precision_score(labels, scores)

        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc)
        
        print("positives: {}, negatives: {}".format(positive, negative))
        print("ROC-AUC: {}, PR-AUC: {}".format(roc_auc, pr_auc))
        # F1_score_plot(labels, scores)
        print("--")

    if save is True:
        index = [test_file.split('/')[-1] for test_file in test_files]
        columns = ['roc-auc', 'pr-auc']
        records_df = pd.DataFrame(index=index, columns=columns)

        for i in range(len(index)):
            records_df['roc-auc'][index[i]] = roc_aucs[i]
            records_df['pr-auc'][index[i]] = pr_aucs[i]
            
        with open(result_file, 'wb') as f:
            pickle.dump(records_df, f)

def get_mse_loss(targets, predicts):
    # targets : [batch, example_length, target_dim]
    # predicts : [batch, example_length, output_dim]
    criterion = nn.MSELoss(reduction='none')
    values = torch.sum(torch.mean(criterion(targets, predicts), dim=2), dim=1)
    
    return values

def F1_score(labels, preds):

    confusion = metrics.confusion_matrix(labels, preds)
    accuracy = metrics.accuracy_score(labels, preds)
    precision = metrics.precision_score(labels, preds)
    recall = metrics.recall_score(labels, preds)

    f1 = metrics.f1_score(labels, preds)

    print(f"TN {confusion[0][0]}\t/ FP {confusion[0][1]}")
    print(f"FN {confusion[1][0]}\t/ TP {confusion[1][1]}")
    print(
        "accuracy: {0:.4f}, precsion: {1:.4f}, recall: {2:.4f}, F1: {3:.4f}".format(
            accuracy, precision, recall, f1
        )
    )
def F1_score_plot(labels, preds):
    import numpy as np
    from sklearn.preprocessing import Binarizer
    thresholds = np.arange(0.1, 1, 0.1)
    preds = np.reshape(preds, (-1, 1))
    for custom_threshold in thresholds:
    
        binarizer = Binarizer(threshold=custom_threshold).fit(preds)
        custom_predict = binarizer.transform(preds)
        print("threshold: ", custom_threshold)
        print("-" * 60)
        F1_score(labels, custom_predict)
        print("=" * 60)    