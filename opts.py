import argparse
parser = argparse.ArgumentParser(description="PyTorch SASRec")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model_name', default='SasRec', type=str,
                    help='Model Name')
parser.add_argument('--n', default=50, type=int,
                    help='length of sequence')
parser.add_argument('--d', default=50, type=int,
                    help='hidden dimension')
parser.add_argument('--n_epochs', default=10, type=int,
                    help='number of epochs')
parser.add_argument('--lr', default=0.001, type=int,
                    help='learning rate')





