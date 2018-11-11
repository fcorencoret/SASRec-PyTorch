from data_loader import MovieLensLoader
from model import SASRec
import torch
from callbacks import AverageMeter
from utils import accuracy, save_checkpoint, multiple_binary_cross_entropy
import time
import torch.nn.functional as F

ROOT_PATH = 'data'
n = 50
d = 300
BATCH_SIZE = 2
lr = 0.001
num_epochs = 1
resume = False
start_epoch = 0
print_freq = 2
stride=None
best_loss = float('Inf')
output_dir = 'checkpoints'
model_name = 'SelfAttention'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
	global best_loss
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
		attention_stack=3,
		ffn_hidden_dim=100,
		dropout=0.2).to(device)
	print(" > Created the model")

	# define optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	print(" > Training is getting started...")
	print(" > Training takes {} epochs.".format(num_epochs))
	for epoch in range(start_epoch, num_epochs):

		# train for one epoch
		train_loss, train_top1, train_top5 = train(train_loader, model, optimizer, epoch)

		# evaluate on validation set
		val_loss, val_top1, val_top10 = validate(val_loader, model)
		print(" > Validation loss after epoch {} = {}".format(epoch, val_loss))


		# remember best loss and save the checkpoint
		is_best = val_loss < best_loss
		best_loss = min(val_loss, best_loss)
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': "SelfAttention",
			'state_dict': model.state_dict(),
			'best_loss': best_loss,
		}, is_best, output_dir, model_name)

def train(train_loader, model, optimizer, epoch):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	model.train()

	end = time.time()
	for i , (input , target) in enumerate(train_loader):

		# measure data loading time
		data_time.update(time.time() - end)

		# reset model gradients
		model.zero_grad()

		# compute output and loss
		loss = torch.zeros(1).to(device)
		output = model(input)
		loss = multiple_binary_cross_entropy(input, target, output, loss)	

		# measure accuracy and record loss
		prec1, prec10 = accuracy(output, target, topk=(1, 10))
		losses.update(loss.item(), input.size(0))
		top1.update(prec1.item(), input.size(0))
		top10.update(prec10.item(), input.size(0))

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
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				'Prec@10 {top10.val:.3f} ({top10.avg:.3f})'.format(
					epoch, i, len(train_loader), batch_time=batch_time,
					data_time=data_time, loss=losses, top1=top1, top10=top10))

	return losses.avg, top1.avg, top10.avg

def validate(val_loader, model, class_to_idx=None):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	with torch.no_grad():
		for i, (input, target) in enumerate(val_loader):
		
			# compute output and loss
			loss = torch.zeros(1).to(device)
			output = model(input)
			loss = multiple_binary_cross_entropy(input, target, output, loss)	

			# measure accuracy and record loss
			prec1, prec10 = accuracy(output, target, topk=(1, 10))
			losses.update(loss.item(), input.size(0))
			top1.update(prec1.item(), input.size(0))
			top10.update(prec10.item(), input.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % print_freq == 0:
				print('Val: [{0}/{1}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'HitRate@1 {top1.val:.3f} ({top1.avg:.3f})\t'
					'HitRate@10 {top10.val:.3f} ({top10.avg:.3f})'.format(
						i, len(val_loader), batch_time=batch_time, loss=losses,
						top1=top1, top10=top10))

	print(' * HitRate@1 {top1.avg:.3f} HitRate@10 {top10.avg:.3f}'
			.format(top1=top1, top10=top10))

	return losses.avg, top1.avg, top10.avg

if __name__ == '__main__':
	main()
