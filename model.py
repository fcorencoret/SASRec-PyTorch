import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import math

class EmbeddingLayer(nn.Module):
	def __init__(self, input_dim, output_dim=300, n=30):
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.n = n
		self.embedding = nn.Embedding(self.input_dim, self.output_dim, padding_idx=0)

	def forward(self, X):
		embed = self.embedding(X.long())
		positional_embedding = autograd.Variable(torch.Tensor(self.n, 1), requires_grad=True)
		nn.init.xavier_uniform_(positional_embedding)
		return embed + positional_embedding



class AttentionBlock(nn.Module):
	def __init__(self, n_items, d=300, n=30, ffn_hidden_dim=100):
		super().__init__()
		self.n_items = n_items
		self.d = d
		self.n = n
		self.ffn_hidden_dim = ffn_hidden_dim
		self.input_embedding = EmbeddingLayer(self.n_items, self.d, self.n)
		self.key_embedding = nn.Linear(self.d, self.d)
		self.query_embedding = nn.Linear(self.d, self.d)
		self.value_embedding = nn.Linear(self.d, self.d)
		self.linear1 = nn.Linear(self.n * self.d, self.ffn_hidden_dim)
		self.linear2 = nn.Linear(self.ffn_hidden_dim, self.n_items)

	def forward(self, X):
		embed_input = self.input_embedding(X)
		embed_key = self.key_embedding(embed_input)
		embed_query = self.query_embedding(embed_input)
		embed_value = self.value_embedding(embed_input)
		query_key_dot = torch.bmm(embed_query, embed_key.permute(0, 2, 1)) /  math.sqrt(self.d)
		attention = torch.bmm(F.softmax(query_key_dot, dim=2), embed_value)
		flatten = attention.view(attention.size(0), -1)
		ffn1 = self.linear1(flatten)
		relu = F.relu(ffn1)
		ffn2 = self.linear2(ffn1)
		softmax = F.softmax(ffn2, dim=1)	
		return ffn2




