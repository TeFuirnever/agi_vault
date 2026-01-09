import torch
import torch.nn as nn

# ----------------------------
# 配置参数
# ----------------------------
vocab_size = 1000    # 假设词表大小
embed_dim = 32       # 词向量维度
num_heads = 4        # 注意力头数
hidden_dim = 64      # 前馈网络隐藏层维度
num_layers = 2       # Transformer 编码器层数
seq_len = 10         # 输入序列长度（词数）
batch_size = 1       # batch大小

# ----------------------------
# 输入示例：随机一句话（词索引）
# ----------------------------
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))  # [batch, seq_len]

# ----------------------------
# Embedding 层
# ----------------------------
embedding = nn.Embedding(vocab_size, embed_dim)
x = embedding(input_ids)  # [batch_size, seq_len, embed_dim]

# PyTorch Transformer 默认输入维度是 [seq_len, batch, embed_dim]，需要转置
x = x.transpose(0, 1)  # [seq_len, batch_size, embed_dim]

# ----------------------------
# Transformer 编码器
# ----------------------------
encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim, 
    nhead=num_heads, 
    dim_feedforward=hidden_dim
)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

# 前向传播
output = transformer_encoder(x)  # [seq_len, batch_size, embed_dim]

# 再转回 [batch_size, seq_len, embed_dim] 方便理解
output = output.transpose(0, 1)

print("输入维度:", input_ids.shape)      # [batch_size, seq_len]
print("输出维度:", output.shape)         # [batch_size, seq_len, embed_dim]
