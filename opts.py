import argparse
parser = argparse.ArgumentParser(description="PyTorch SASRec")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--n_epochs', default=10, type=int,
                    help='number of epochs')




