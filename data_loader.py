import torch.utils.data as data
import preprocess_dataset
import json
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MovieLensLoader(data.Dataset):
	def __init__(self, root_path, dataset='train', n=50, stride=5):
		self.root_path = root_path
		self.dataset = dataset
		self.n = n
		self.stride = stride
		self._load_data()
		self.current_user = 1
		self.current_item = len(self.user_train[self.current_user]) - self.n - 1

	def _load_data(self):
		with open('%s/user_train.json'%self.root_path, 'r') as infile:
			self.user_train = json.load(infile)
			self.user_train = {int(k):[int(i) for i in v] for k,v in self.user_train.items()}
			self.total_data = 0
			if self.dataset == 'train':
				for user in self.user_train.values():
					self.total_data += max(1, ((len(user) - self.n) // self.stride) + 1)
			if self.dataset == 'val' or self.dataset == 'test':
				self.total_data = len(self.user_train)

		if self.dataset == 'val' or self.dataset == 'test':
			with open('%s/user_val.json'%self.root_path, 'r') as infile:
				self.user_val = json.load(infile)
				self.user_val = {int(k):[int(i) for i in v] for k,v in self.user_val.items()}
	
		if self.dataset == 'test':
			with open('%s/user_test.json'%self.root_path, 'r') as infile:
				self.user_test = json.load(infile)
				self.user_test = {int(k):[int(i) for i in v] for k,v in self.user_test.items()}

		with open('%s/usernum.txt'%self.root_path, 'r') as infile:
			self.usernum = int(infile.readline().rstrip())

		with open('%s/itemnum.txt'%self.root_path, 'r') as infile:
			self.itemnum = int(infile.readline().rstrip())		

	def __getitem__(self, index):
		if self.dataset == 'val' or self.dataset == 'test': return self._getitem_val_test(index)

		user_reviews = self.user_train[self.current_user]
		n_user_reviews = len(user_reviews)

		if self.current_item < 0:
			x = torch.zeros(self.n).long()		
			y = torch.zeros(self.n).long()		
			n_ratings = len(self.user_train[self.current_user])
			
			# window slide option
			if n_ratings >= self.n + 1: 
				n_ratings = self.n + self.current_item	
				x[self.n - n_ratings + 1: ] = torch.LongTensor(user_reviews[:n_ratings - 1])	
				y[self.n - n_ratings + 1: ] = torch.LongTensor(user_reviews[1 :n_ratings])

			else: 
				x[self.n - n_ratings + 1: ] = torch.LongTensor(user_reviews[:-1])	
				y[self.n - n_ratings + 1: ] = torch.LongTensor(user_reviews[1:])	

			if self.current_user == self.usernum: self._reset_current_indices()
			else:		
				self.current_user += 1
				self.current_item = len(self.user_train[self.current_user]) - self.n - 1
			return (x.to(device), y.to(device))

		elif self.current_item > 0:
			x = torch.LongTensor(user_reviews[self.current_item: self.current_item + self.n])
			y = torch.LongTensor(user_reviews[self.current_item + 1: self.current_item + self.n + 1])
			self.current_item -= self.stride
			return (x.to(device), y.to(device))

		elif self.current_item == 0:
			x = torch.LongTensor(user_reviews[self.current_item: self.current_item + self.n])
			y = torch.LongTensor(user_reviews[self.current_item + 1: self.current_item + self.n + 1])
			if self.current_user == self.usernum: self._reset_current_indices()
			else:
				self.current_user += 1
				self.current_item = len(self.user_train[self.current_user]) - self.n - 1
			return (x.to(device), y.to(device))


	def _getitem_val_test(self, index):
		idx = index + 1
		x = torch.zeros(self.n).long()
		y = torch.zeros(self.n).long()
		n_ratings = len(self.user_train[idx])
		if n_ratings >= self.n + 1:
			x = torch.LongTensor(self.user_train[idx][n_ratings - self.n - 1 : -1])
			y = torch.LongTensor(self.user_train[idx][n_ratings - self.n: ])
		else:
			x[self.n - n_ratings + 1: ] = torch.LongTensor(self.user_train[idx][:-1])
			x[self.n - n_ratings + 1: ] = torch.LongTensor(self.user_train[idx][1:])
		

		if self.dataset == 'val': 
			x[:-1] = x[1:]
			x[-1] = self.user_train[idx][-1]
			y = self.user_val[idx][0]

		if self.dataset == 'test': 
			x[:-2] = x[2:]
			x[-2] = self.user_train[idx][-1]
			x[-1] = self.user_val[idx][0]
			y = self.user_test[idx][0]

		return (x, y)

	def _reset_current_indices(self):
		self.current_user = 1
		self.current_item = len(self.user_train[self.current_user]) - self.n - 1


	def __len__(self):
		return self.total_data

