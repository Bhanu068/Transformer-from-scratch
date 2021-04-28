import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert(self.head_dim * heads == embed_size)
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias =  False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)
        
    def forward(self, values, keys, queries, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.values(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        queries = self.queries(queries)
        keys = self.keys(keys)
        
        energy = torch.einsum("nqhd, nkhd -->nhqk", [queries, keys]) 
        # Queries shape ---> (N, query_len, heads, head_dim)
        # Keys shape ------> (N, key_len, heads, head_dim)
        # energy shape ----> (N, hedas, query_len, key_len)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            
        attention = torch.softmax(energy / (self.embed_size ** ( 1 / 2)), dim = 3)
        # attention shape ---> (N, heads, query_len, key_len)
        # values shape ------> (N, value_len, heads, head_dim)
        # out shape ---------> (N, query_len, heads, head_dim)
        out = torch.einsum("nhqk, nvhd --> nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.src_vocab_size = src_vocab_size
        self.num_layers = num_layers
        self.heads = heads
        self.device = device
        self.forward_expansion = forward_expansion
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_encoding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)
            ]
        )
    
    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_encoding(positions))
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, value, key, src_mask, tgt_mask):
        attention = self.attention(x, x, x, tgt_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out
    
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.device = device
        self.word_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.position_encoding = nn.Embedding(embed_size, embed_size)
        
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)
            ]
        )
        
        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask, tgt_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_encoding(positions))
        
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        
        out = self.fc_out(x)
        
        return out
        

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx, embed_size = 512, num_layers = 6,
                 forward_expansion = 4, heads = 8, dropout = 0, device = 'cuda', max_length = 100):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(tgt_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)
        
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_length)
        return src_mask.to(self.device)
    
    def make_tgt_mask(self, tgt):
        N, tgt_len = tgt.shape
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(N, 1, tgt_len, tgt_len)
        
        return tgt_mask.to(device)
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(tgt, enc_src, src_mask, tgt_mask)
        return out