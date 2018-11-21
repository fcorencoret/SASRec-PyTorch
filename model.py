import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class EmbeddingLayer(nn.Module):
	def __init__(self, n_items, d=300, n=50, scale=True, pos_enc=False):
		super().__init__()
		self.n_items = n_items if pos_enc else n_items + 1
		self.d = d
		self.n = n
		self.embedding = nn.Embedding(self.n_items, self.d, padding_idx=None if pos_enc else 0).to(device)
		self.scale = scale

	def forward(self, X):
		# Calculate Embedding
		output = self.embedding(X.long())
		
		# Apply scaling 
		if self.scale: output = output * (self.n ** 0.5) 

		return output

class PositionalEncoding(nn.Module):
	def __init__(self):
		super().__init__()
		# TODO 
		# Add Sinusoidal positional encoding

class AttentionBlock(nn.Module):
	def __init__(self, n_items, d=300, n=50, ffn_hidden_dim=50, dropout=0.2):
		super().__init__()
		self.n_items = n_items
		self.d = d
		self.n = n
		self.ffn_hidden_dim = ffn_hidden_dim
		# TODO test without bias
		self.key_embedding = nn.Linear(self.d, self.d, bias=True).to(device)
		self.query_embedding = nn.Linear(self.d, self.d, bias=True).to(device)
		self.value_embedding = nn.Linear(self.d, self.d, bias=True).to(device)
		self.dropout = nn.Dropout(dropout).to(device)
		self.pointwise = PointWiseFFN(self.dropout, self.ffn_hidden_dim)
		

	def forward(self, queries, keys, padding_mask):
		# Calculate Self Attention
		embed_query = self.query_embedding(queries)
		embed_key = self.key_embedding(keys)
		embed_value = self.value_embedding(keys)
		attention_coefs = torch.bmm(embed_query, embed_key.permute(0, 2, 1)) /  (self.d ** 0.5)

		# Apply Key Masking 
		key_padding_mask = ((torch.sum(keys, dim=-1) == 0).to(device, dtype=torch.float32) * (-2**32 + 1)).unsqueeze(1).repeat(1, self.n, 1)
		output = torch.where(padding_mask.unsqueeze(1).repeat(1, self.n, 1).eq(0), key_padding_mask, attention_coefs) 

		# Activations
		output = F.softmax(output, dim=2)

		# Query Masking
		query_padding_mask = (torch.sum(queries, dim=-1) != 0).to(device, dtype=torch.float32).unsqueeze(1).repeat(1, self.n, 1)
		output = output * query_padding_mask

		# Dropout
		output = self.dropout(output)

		# Weighted sum
		output = torch.bmm(output, embed_value)

		# Residual Connection
		output = output + queries

		return output

class PointWiseFFN(nn.Module):
	def __init__(self, dropout, ffn_hidden_dim=50):
		super().__init__()
		self.dropout = dropout
		self.ffn_hidden_dim = ffn_hidden_dim
		self.ffn1 = nn.Conv1d(self.ffn_hidden_dim, self.ffn_hidden_dim, 1, bias=True).to(device)
		self.ffn2 = nn.Conv1d(self.ffn_hidden_dim, self.ffn_hidden_dim, 1, bias=True).to(device)

	def forward(self, X):
		# First PointWise Feed-Forward Network 
		ffn1 = self.ffn1(X)
		ffn1 = F.relu(ffn1)
		ffn1 = self.dropout(ffn1)

		# Second PointWise Feed-Forward Network
		ffn2 = self.ffn2(ffn1)
		ffn2 = self.dropout(ffn2)
		return X + ffn2

class SASRec(nn.Module):
	def __init__(self, n_items, d=300, n=30, attention_stack=2, ffn_hidden_dim=50, dropout=0.2):
		super().__init__()
		self.n_items = n_items
		self.d = d
		self.n = n
		self.ffn_hidden_dim = ffn_hidden_dim

		self.sequence_embedding = EmbeddingLayer(self.n_items, self.d, self.n)
		self.positional_encoding_embedding = EmbeddingLayer(self.n, self.d, self.n, scale=False, pos_enc=True)
		self.dropout = nn.Dropout(dropout).to(device)
		self.attn_norm = nn.LayerNorm(self.d).to(device)
		self.ffn_norm = nn.LayerNorm(self.d).to(device)
		self.norm = nn.LayerNorm(self.d).to(device)

		self.attention_stack = []
		for i in range(attention_stack):
			self.attention_stack.append((
				AttentionBlock(
					n_items=self.n_items, 
					d=self.d, 
					n=self.n,
					ffn_hidden_dim=self.ffn_hidden_dim),
				PointWiseFFN(
					self.dropout,
					ffn_hidden_dim=self.ffn_hidden_dim)
			))

	# def get_item_embedding_table(self):
		# all_items = torch.tensor([i for i in range(self.n_items + 1)], dtype=torch.long).to(device)
		# return self.sequence_embedding.embedding(all_items).to(device)

	def embedding_lookup(self, seq):
		return self.sequence_embedding.embedding(seq).to(device)


	def calculate_embedding_distances(self, batches):		
		relevances = torch.zeros((batches.size(0), batches.size(1), self.all_items_embeddings.size(0))).to(device)
		for index, batch in enumerate(batches):
			relevances[index] = torch.mm(batch, self.all_items_embeddings.t())
		return relevances

	def forward(self, seq, pos, neg):
		# Sequence Embedding
		output = self.sequence_embedding(seq)
		
		# Positional Encoding Embedding
		pos_enc = self.positional_encoding_embedding(torch.Tensor([i for i in range(self.n)]).to(device))
		output = output + pos_enc

		# Apply dropout
		output = self.dropout(output)

		# Apply Padding mask
		padding_mask = (seq != 0).to(device, dtype=torch.float32)
		output = output * padding_mask.unsqueeze(-1).expand_as(output)

		# Attention Blocks
		for (attention_block, point_wise) in self.attention_stack:
			output = attention_block(self.attn_norm(output), output, padding_mask)
			output = point_wise(self.ffn_norm(output))
			output = output * padding_mask.unsqueeze(-1).expand_as(output)

		# Normalization 
		seq_emb = self.norm(output)

		# Expand to size (batch_size * self.n, self.d) for later dot product
		seq_emb = seq_emb.view(seq_emb.size()[0] * seq_emb.size()[1], -1)
		pos = pos.view(pos.size()[0] * pos.size()[1])
		neg = neg.view(neg.size()[0] * neg.size()[1])
		
		# Embedding lookup from item_emb_table
		pos_emb = self.embedding_lookup(pos)
		neg_emb = self.embedding_lookup(neg)
		return seq_emb, pos_emb, neg_emb



