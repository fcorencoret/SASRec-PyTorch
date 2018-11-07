import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class EmbeddingLayer(nn.Module):
	def __init__(self, n_items, d=300, n=30, dropout=0.2):
		super().__init__()
		self.n_items = n_items
		self.d = d
		self.n = n
		self.embedding = nn.Embedding(self.n_items, self.d, padding_idx=0).to(device)
		self.dropout = nn.Dropout(dropout).to(device)

	def forward(self, X):
		embed = self.embedding(X.long())
		return self.dropout(embed)
                #positional_embedding = autograd.Variable(torch.Tensor(self.n, 1), requires_grad=True)
		#nn.init.xavier_uniform_(positional_embedding)
		#return self.dropout(embed + positional_embedding)



class AttentionBlock(nn.Module):
	def __init__(self, n_items, d=300, n=30, ffn_hidden_dim=100, dropout=0.2):
		super().__init__()
		self.n_items = n_items
		self.d = d
		self.n = n
		self.ffn_hidden_dim = ffn_hidden_dim
		self.key_embedding = nn.Linear(self.d, self.d, bias=False).to(device)
		self.query_embedding = nn.Linear(self.d, self.d, bias=False).to(device)
		self.value_embedding = nn.Linear(self.d, self.d, bias=False).to(device)
		self.linear1 = nn.Linear(self.d, self.d, bias=True).to(device)
		self.linear2 = nn.Linear(self.d, self.d, bias=True).to(device)
		self.normalize = nn.LayerNorm((self.n, self.d)).to(device)
		self.dropout = nn.Dropout(dropout).to(device)
		

	def forward(self, X):
		norm = self.normalize(X)
		embed_key = self.key_embedding(norm)
		embed_query = self.query_embedding(norm)
		embed_value = self.value_embedding(norm)
		query_key_dot = torch.bmm(embed_query, embed_key.permute(0, 2, 1)) /  math.sqrt(self.d)
		attention = torch.bmm(F.softmax(query_key_dot, dim=2), embed_value)
		ffn1 = self.linear1(attention)
		ffn2 = self.linear2(ffn1)
		return X + self.dropout(ffn2)

class SASRec(nn.Module):
	def __init__(self, n_items, d=300, n=30, attention_stack=2, ffn_hidden_dim=100, dropout=0.2):
		super().__init__()
		self.n_items = n_items
		self.d = d
		self.n = n
		self.ffn_hidden_dim = ffn_hidden_dim

		self.input_embedding = EmbeddingLayer(self.n_items, self.d, self.n)

		self.attention_stack = []
		for i in range(attention_stack):
			self.attention_stack.append(AttentionBlock(
				n_items=self.n_items, 
				d=self.d, 
				n=self.n))

		self.relevance = nn.Linear(self.d, self.n_items, bias=False).to(device)

	def forward(self, X):
		out = self.input_embedding(X)
		for attention_block in self.attention_stack:
			out = attention_block(out)
		out = self.relevance(out)
		# softmax = F.softmax(out, dim=1)
		return out



