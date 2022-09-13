import argparse
import torch
import random
import numpy as np

from core.configuration import CONFIGS
# from utils.wrapper import train_model, test_model
from utils.wrapper_prediction_quantile import train_model, test_model
def main(args):
    #set seed#
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = CONFIGS[args.dataset]()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.EPOCHS = 100
    args.batch_size = 64
    args.state_file = '/home/hjjung/ailab/anomaly_TFT/model/WADI/prediction/model_epochs_{}_hidden_{}_ex_{}.pt'.format(args.EPOCHS, config.hidden_size, config.example_length)

    if args.MODE == 'train':
        if args.dataset == 'GHL':
            train_file = '/home/hjjung/ailab/anomaly_TFT/templates/GHL/train_1500000_seed_11_vars_23.csv'
        elif args.dataset == 'WADI':
            train_file = '/home/hjjung/ailab/anomaly_TFT/templates/WADI/WADI_Non-null_Date.csv'
        elif args.dataset == 'SMD':
            train_file = 'data/SMD_train.csv'
        train_model(train_file, args, config)

    elif args.MODE == 'test':
        if args.dataset == 'GHL':
            test_file = '/home/hjjung/ailab/anomaly_TFT/templates/GHL/test/'
        elif args.dataset == 'WADI':
            test_file = '/home/hjjung/ailab/anomaly_TFT/templates/WADI/test/'
        elif args.dataset == 'SMD':
            test_file = 'data/test/'
        test_model(test_file, args, config, save=True, attention=True)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', dest="dataset", type=str, required=True,
                        choices=CONFIGS.keys())
    parser.add_argument('--mode', dest="MODE", type=str, required=True,
                        help='run mode: [train|test]')
    parser.add_argument('--epochs', dest="EPOCHS", type=int, default=10)
    
    args = parser.parse_args()

    return args

if __name__=='__main__':
    args = parse_arguments()
    main(args)