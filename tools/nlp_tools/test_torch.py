import torch
from torch import Tensor
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)



# Implementing Sparse Transformers

# Sparse Transformers are a variant of the standard Transformer architecture
# that use sparse attention patterns to reduce the quadratic complexity of attention.
# This allows them to handle longer sequences more efficiently.

# Let's implement a simple sparse attention mechanism

class SparseAttention(torch.nn.Module):
    def __init__(self, block_size=16, num_heads=8, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.num_heads = num_heads
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        # query, key, value shape: [batch_size, seq_len, embedding_dim]
        batch_size, seq_len, embedding_dim = query.shape
        head_dim = embedding_dim // self.num_heads
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # Create sparse attention mask (block-sparse pattern)
        # This is a simplified version - real sparse transformers use more complex patterns
        sparse_mask = torch.zeros_like(scores, dtype=torch.bool)
        
        # Create block diagonal pattern
        for i in range(0, seq_len, self.block_size):
            end_idx = min(i + self.block_size, seq_len)
            sparse_mask[:, :, i:end_idx, i:end_idx] = True
            
        # Apply the sparse mask
        if mask is not None:
            sparse_mask = sparse_mask & mask
            
        # Set attention scores for masked positions to -inf
        scores = scores.masked_fill(~sparse_mask, float('-inf'))
        
        # Apply softmax and dropout
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Get the output
        output = torch.matmul(attention_weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embedding_dim)
        
        return output

# Example of a Sparse Transformer Block
class SparseTransformerBlock(torch.nn.Module):
    def __init__(self, embedding_dim=512, num_heads=8, ff_dim=2048, dropout=0.1, block_size=16):
        super().__init__()
        self.attention = SparseAttention(block_size=block_size, num_heads=num_heads, dropout=dropout)
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.norm2 = torch.nn.LayerNorm(embedding_dim)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, ff_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ff_dim, embedding_dim),
            torch.nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attention_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

# Test the sparse transformer with a small example
seq_len = 64
batch_size = 2
embedding_dim = 512

# Create random input
input_tensor = torch.rand(batch_size, seq_len, embedding_dim)

# Initialize the sparse transformer block
sparse_transformer = SparseTransformerBlock(embedding_dim=embedding_dim, block_size=16)

# Forward pass
output = sparse_transformer(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
print("Sparse Transformer successfully implemented!")

# Note: This is a simplified implementation. Real sparse transformers like those in 
# "Generating Long Sequences with Sparse Transformers" (Child et al., 2019) use more 
# complex sparsity patterns such as strided and fixed patterns to capture both local 
# and global dependencies while maintaining computational efficiency.
