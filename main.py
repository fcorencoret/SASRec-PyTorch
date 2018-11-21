from data_loader import MovieLensLoader
from model import SASRec
import torch
from callbacks import AverageMeter
from utils import accuracy, save_checkpoint, multiple_binary_cross_entropy, plot
import time
import torch.nn.functional as F
from opts import parser
import os
import numpy as np

ROOT_PATH = 'data'
n = 50
d = 50
BATCH_SIZE = 16 if torch.cuda.is_available() else 2
lr = 0.001
num_epochs = 10 if torch.cuda.is_available() else 1
start_epoch = 0
print_freq = 2
stride=None
best_loss = float('Inf')
output_dir = 'checkpoints'
model_name = 'SelfAttention'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
store = {
	'train_loss' : [],
	'val_loss' : [],
	'train_hitrate@1' : [],
	'train_hitrate@10' : [],
	'train_ndcg@10' : [],
	'val_hitrate@1' : [],
	'val_hitrate@10' : [],
	'val_ndcg@10' : [],
}


def main():
	global args, best_loss, start_epoch
	args = parser.parse_args()
	if args.n_epochs: num_epochs = args.n_epochs

	# load data
	train_loader = torch.utils.data.DataLoader(
		MovieLensLoader(ROOT_PATH, 
			dataset='train',
			n=n,
			stride=stride),
		batch_size=BATCH_SIZE)

	val_loader = torch.utils.data.DataLoader(
		MovieLensLoader(ROOT_PATH, 
			dataset='val',
			n=n),
		batch_size=BATCH_SIZE)

	print(" > Loaded the data")
	print(" > Train length: {}".format(len(train_loader)))
	print(" > Val length: {}".format(len(val_loader)))

	# create model
	model = SASRec(
		n_items=train_loader.dataset.itemnum, 
		d=d, 
		n=n,
		attention_stack=2,
		ffn_hidden_dim=n,
		dropout=0.2).to(device)
	print(" > Created the model")

	# resume
	if args.resume:
		if os.path.isfile(args.resume):
			print(("=> loading checkpoint '{}'".format(args.resume)))
			checkpoint = torch.load(args.resume)
			start_epoch = checkpoint['epoch']
			num_epochs += start_epoch

			# Load data from previous training
			with open(os.path.join(output_dir, model_name) + '_data.json', 'r') as fp:
				store = json.load(fp)
			# best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint (epoch {})"
			      .format(checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	# define optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	print(" > Training is getting started...")
	print(" > Training takes {} epochs.".format(num_epochs))
	

	for epoch in range(start_epoch, num_epochs):

		# train for one epoch
		train_loss, train_top1, train_top10, train_nDCG10 = train(train_loader, model, optimizer, epoch)

		# evaluate on validation set
		val_loss, val_top1, val_top10, val_nDCG10 = validate(val_loader, model)
		print(" > Validation loss after epoch {} = {}".format(epoch, val_loss))


		# store train and val loss
		store['train_loss'].append(train_loss)
		store['val_loss'].append(val_loss)
		store['train_hitrate@1'].append(train_top1)
		store['train_hitrate@10'].append(train_top10)
		store['train_ndcg@10'].append(train_nDCG10)
		store['val_hitrate@1'].append(val_top1)
		store['val_hitrate@10'].append(val_top10)
		store['val_ndcg@10'].append(val_nDCG10)

		# remember best loss and save the checkpoint
		is_best = val_loss < best_loss
		best_loss = min(val_loss, best_loss)
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': "SelfAttention",
			'state_dict': model.state_dict(),
			'best_loss': best_loss,
		}, is_best, output_dir, model_name)

	# plot training results
	plot(store, output_dir, model_name)

def train(train_loader, model, optimizer, epoch):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top10 = AverageMeter()
	nDCG10 = AverageMeter()

	model.train()

	end = time.time()
	for i , (seq , pos, neg) in enumerate(train_loader):

		# measure data loading time
		data_time.update(time.time() - end)

		# reset model gradients
		model.zero_grad()

		# compute output and loss
		seq_emb, pos_emb, neg_emb = model(seq, pos, neg)
		loss = multiple_binary_cross_entropy(seq_emb, pos_emb, pos, neg_emb).to(device)	
		# measure accuracy and record loss
		prec1, prec10, nDCG = accuracy(seq_emb, pos, topk=(1, 10))
		losses.update(loss.item(), seq.size(0))
		top1.update(prec1.item(), seq.size(0))
		top10.update(prec10.item(), seq.size(0))
		nDCG10.update(nDCG, seq.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				'HitRate@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				'HitRate@10 {top10.val:.3f} ({top10.avg:.3f})\t'
				'nDCG@10 {nDCG10.val:.3f} ({nDCG10.avg:.3f})'.format(
					epoch, i, len(train_loader), batch_time=batch_time,
					data_time=data_time, loss=losses, top1=top1, top10=top10, nDCG10=nDCG10))

	return losses.avg, top1.avg, top10.avg, nDCG10.avg

def validate(val_loader, model, class_to_idx=None):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top10 = AverageMeter()
	nDCG10 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	with torch.no_grad():
		for i, (seq, pos, neg) in enumerate(val_loader):
		
			# compute output and loss
			seq_emb, pos_emb, neg_emb = model(seq, pos, neg)
			loss = multiple_binary_cross_entropy(seq_emb, pos_emb, pos, neg_emb).to(device)	

			# measure accuracy and record loss
			prec1, prec10, nDCG = accuracy(output, pos, topk=(1, 10))
			losses.update(loss.item(), seq.size(0))
			top1.update(prec1.item(), seq.size(0))
			top10.update(prec10.item(), seq.size(0))
			nDCG10.update(nDCG, seq.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % print_freq == 0:
				print('Val: [{0}/{1}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'HitRate@1 {top1.val:.3f} ({top1.avg:.3f})\t'
					'HitRate@10 {top10.val:.3f} ({top10.avg:.3f})\t'
					'nDCG@10 {nDCG10.val:.3f} ({nDCG10.avg:.3f})'.format(
						i, len(val_loader), batch_time=batch_time, loss=losses,
						top1=top1, top10=top10, nDCG10=nDCG10))

	print(' * HitRate@1 {top1.avg:.3f} - HitRate@10 {top10.avg:.3f} * nDCG@10 {nDCG10.avg:.3f}'
			.format(top1=top1, top10=top10, nDCG10=nDCG10))

	return losses.avg, top1.avg, top10.avg, nDCG10.avg

import signal
import sys
def signal_handler(sig, frame):
	plot(store, output_dir, model_name)
	sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)



if __name__ == '__main__':
	main()
