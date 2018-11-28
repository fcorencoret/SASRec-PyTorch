from data_loader import MovieLensLoader
from model import SASRec
import torch
from callbacks import AverageMeter
from utils import accuracy, save_checkpoint, multiple_binary_cross_entropy, plot
import time
import torch.nn.functional as F
from opts import parser
import os
import json

ROOT_PATH = 'data'
n = 50
d = 50
BATCH_SIZE = 128 if torch.cuda.is_available() else 2
lr = 0.001
beta2 = 0.98
num_epochs = 350 if torch.cuda.is_available() else 1
start_epoch = 0
print_freq = 2
eval_freq = 20
b = 2
stride = None
best_hitrate10 = 0
output_dir = 'checkpoints'
model_name = 'SasRec'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
store = {
	'train_loss' : [],
	'train_hitrate@1' : [],
	'train_hitrate@10' : [],
	'train_ndcg@10' : [],
	'val_hitrate@1' : [],
	'val_hitrate@10' : [],
	'val_ndcg@10' : [],
}


def main():
	global args, best_hitrate10, start_epoch, store, model_name
	args = parser.parse_args()
	num_epochs = args.n_epochs
	model_name = args.model_name
	lr = args.lr
	n = args.n
	b = args.b

	# load data
	train_loader = torch.utils.data.DataLoader(
		MovieLensLoader(ROOT_PATH, 
			dataset='train',
			n=n,
			stride=stride),
		batch_size=BATCH_SIZE)

	# train eval
	train_eval = torch.utils.data.DataLoader(
		MovieLensLoader(ROOT_PATH, 
			dataset='train_eval',
			n=n),
		batch_size=1)

	# val eval
	val_eval = torch.utils.data.DataLoader(
		MovieLensLoader(ROOT_PATH, 
			dataset='val_eval',
			n=n),
		batch_size=1)

	# test eval
	test_eval = torch.utils.data.DataLoader(
		MovieLensLoader(ROOT_PATH, 
			dataset='test_eval',
			n=n),
		batch_size=1)



	print(" > Loaded the data")
	print(" > Train length: {}".format(len(train_loader)))
	print(" > Train Eval length: {}".format(len(train_eval)))
	print(" > Val Eval length: {}".format(len(val_eval)))
	print(" > Test Eval length: {}".format(len(test_eval)))

	# create model
	model = SASRec(
		n_items=train_loader.dataset.itemnum, 
		d=d, 
		n=n,
		attention_stack=b,
		ffn_hidden_dim=n,
		dropout=0.2).to(device)
	print(" > Created the model")

	# Test modality
	if args.test:
		if os.path.isfile(args.test):
			print(("=> loading checkpoint '{}'".format(args.test)))
			checkpoint = torch.load(args.test)
			epoch = checkpoint['epoch']
			model.load_state_dict(checkpoint['state_dict'])
			test_top1, test_top10, test_nDCG10 = evaluate(test_eval, model, epoch, 'Test')
		else:
			print("=> no checkpoint found at '{}'".format(args.test))
		sys.exit(0)


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
			best_hitrate10 = store['val_hitrate@10'][-1]
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint (epoch {})"
			      .format(checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	# define optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, beta2))

	print(" > Training is getting started...")
	print(" > Training takes {} epochs.".format(num_epochs))

	for epoch in range(start_epoch, num_epochs):

		# train for one epoch
		train_loss = train(train_loader, model, optimizer, epoch)

		# Evaluate the model
		if epoch % eval_freq == 0:
			train_top1, train_top10, train_nDCG10 = evaluate(train_eval, model, epoch, 'Training')
			val_top1, val_top10, val_nDCG10 = evaluate(val_eval, model, epoch, 'Validation')

			# store train and val loss
			store['train_loss'].append(train_loss)
			store['train_hitrate@1'].append(train_top1)
			store['train_hitrate@10'].append(train_top10)
			store['train_ndcg@10'].append(train_nDCG10.item())
			store['val_hitrate@1'].append(val_top1)
			store['val_hitrate@10'].append(val_top10)
			store['val_ndcg@10'].append(val_nDCG10.item())

			# remember best loss and save the checkpoint
			is_best = val_top10 > best_hitrate10
			best_hitrate10 = max(val_top10, best_hitrate10)
			save_checkpoint({
				'epoch': epoch + 1,
				'arch': "SelfAttention",
				'state_dict': model.state_dict(),
				'best_hitrate10': best_hitrate10,
			}, is_best, output_dir, model_name)

			# Early Stopping
			# if not is_best: break


	# Evaluate on three sets
	train_top1, train_top10, train_nDCG10 = evaluate(train_eval, model, epoch, 'Training')
	val_top1, val_top10, val_nDCG10 = evaluate(val_eval, model, epoch, 'Validation')
	test_top1, test_top10, test_nDCG10 = evaluate(test_eval, model, epoch, 'Test')
	# plot training results
	plot(store, output_dir, model_name)

	# remember best loss and save the checkpoint
	is_best = val_top10 > best_hitrate10
	best_hitrate10 = max(val_top10, best_hitrate10)
	save_checkpoint({
		'epoch': epoch + 1,
		'arch': "SelfAttention",
		'state_dict': model.state_dict(),
		'best_hitrate10': best_hitrate10,
	}, is_best, output_dir, model_name)

def train(train_loader, model, optimizer, epoch):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()

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
		losses.update(loss.item(), seq.size(0))
		
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
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
					epoch, i, len(train_loader), batch_time=batch_time,
					data_time=data_time, loss=losses))

	return losses.avg

def evaluate(data_eval, model, epoch, eval):
	top1 = AverageMeter()
	top10 = AverageMeter()
	nDCG10 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	with torch.no_grad():
		for i, (seq, item_idx) in enumerate(data_eval):
			# compute output and loss
			seq_emb, test_emb = model(seq, item_idx, predict=True)
			test_logits = torch.matmul(seq_emb, test_emb.t())
			test_logits = -test_logits.view(seq.size()[0], seq.size()[1], 101)[:, -1, :][0]
			prec1, prec10, nDCG = accuracy(test_logits)

			# update metrics
			top1.update(prec1, 1)
			top10.update(prec10, 1)
			nDCG10.update(nDCG, 1)

	print('-- {eval} Results Epoch [{epoch}] -- \t'
			'* HitRate@1 {top1.avg:.3f} - HitRate@10 {top10.avg:.3f} * nDCG@10 {nDCG10.avg:.3f}'
			.format(eval=eval, epoch=epoch, top1=top1, top10=top10, nDCG10=nDCG10))
	return top1.avg, top10.avg, nDCG10.avg

import signal
import sys
def signal_handler(sig, frame):
	plot(store, output_dir, model_name)
	sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)



if __name__ == '__main__':
	main()
