import torch
import torch.nn as nn
import xxhash

class UnifiedEmbedding(nn.Module):
    def __init__(self, emb_levels, emb_dim):
        super(UnifiedEmbedding, self).__init__()
        self.embedding = nn.Embedding(emb_levels, emb_dim)
        
    def forward(self, x, fnum):
        x_ = torch.LongTensor(x.shape[0], len(fnum))
        for i in range(x.shape[0]):
            for j, h_seed in enumerate(fnum):
                x_[i, j] = xxhash.xxh32(x[i], h_seed).intdigest() % self.embedding.num_embeddings
        return self.embedding(x_).reshape(x_.shape[0], -1)
        