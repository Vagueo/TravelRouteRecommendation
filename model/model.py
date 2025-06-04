import torch
import torch.nn as nn
from config import drop_prob
class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super(MultiHeadAttentionPooling, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.query_vector = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [B, L, D], attention_mask: [B, L]
        B = hidden_states.size(0)
        query = self.query_vector.expand(B, 1, -1)  # [B, 1, D]
        attn_output, _ = self.attention(query, hidden_states, hidden_states,
                                        key_padding_mask=~attention_mask.bool())
        pooled_output = attn_output.squeeze(1)  # [B, D]
        return self.activation(self.linear(pooled_output))

class DualLayerRecModel(nn.Module):
    def __init__(self, mdd_vocab_size, poi_vocab_size, embedding_dim, num_heads, num_layers, bert_out_dim):
        super(DualLayerRecModel, self).__init__()
        self.bert_out_dim = bert_out_dim
        hidden_dim = embedding_dim + bert_out_dim

        # Embedding layers
        self.mdd_embedding = nn.Embedding(mdd_vocab_size, embedding_dim)
        self.poi_embedding = nn.Embedding(poi_vocab_size, embedding_dim)
        self.type_embedding = nn.Embedding(2, embedding_dim)    # 0=MDD, 1=POI
        self.domain_embedding = nn.Embedding(2, embedding_dim)  # 0=MFW, 1=FS

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(drop_prob)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                                   dim_feedforward=embedding_dim * 4, dropout=drop_prob,
                                                   activation='relu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads
        self.output_mdd = nn.Linear(hidden_dim, mdd_vocab_size)
        self.output_poi = nn.Linear(hidden_dim, poi_vocab_size)
        self.output_type = nn.Linear(hidden_dim, 2)

    def forward(self, src, src_types, cls_features, domain_labels):
        """
        src: [B, L]         - 地点ID序列
        src_types: [B, L]   - 地点类型（0=MDD，1=POI）
        cls_features: [B, L, bert_out_dim] - 每个位置的BERT表示
        domain_labels: [B]  - 每个样本的来源（0=MFW，1=生成的MFW数据）
        """
        B, L = src.shape
        device = src.device

        # 获取ID嵌入
        embeddings = torch.zeros(B, L, self.mdd_embedding.embedding_dim).to(device)
        mdd_mask = (src_types == 0)
        poi_mask = (src_types == 1)
        embeddings[mdd_mask] = self.mdd_embedding(src[mdd_mask])
        embeddings[poi_mask] = self.poi_embedding(src[poi_mask])

        # 加入类型 + domain embedding
        type_embeds = self.type_embedding(src_types)
        domain_embeds = self.domain_embedding(domain_labels.unsqueeze(1).repeat(1, L))
        embeddings += type_embeds + domain_embeds

        # 拼接 BERT 特征 + norm + dropout
        combined_embeddings = torch.cat([embeddings, cls_features], dim=-1)  # [B, L, D+H]
        x = self.norm(self.dropout(combined_embeddings))  # [B, L, hidden_dim]

        # Transformer 编码器
        encoded = self.encoder(x)  # [B, L, hidden_dim]

        # 输出层
        mdd_logits = self.output_mdd(encoded)    # [B, L, mdd_vocab_size]
        poi_logits = self.output_poi(encoded)    # [B, L, poi_vocab_size]
        type_logits = self.output_type(encoded)  # [B, L, 2]

        return mdd_logits, poi_logits, type_logits

# class MultiHeadAttentionPooling(nn.Module):
#     def __init__(self, hidden_size, num_heads=8):
#         super(MultiHeadAttentionPooling, self).__init__()
#         self.num_heads = num_heads
#         self.hidden_size = hidden_size
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
#         self.linear = nn.Linear(hidden_size, hidden_size)
#         self.activation = nn.Tanh()
#
#     def forward(self, hidden_states, attention_mask):
#         # hidden_states: [batch_size, seq_len, hidden_size]
#         # attention_mask: [batch_size, seq_len]
#         attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states, key_padding_mask=~attention_mask.bool())
#         # 对注意力输出进行平均池化
#         pooled_output = attn_output.mean(dim=1)
#         return self.activation(self.linear(pooled_output))

# class DualLayerRecModel(nn.Module):
#     def __init__(self, mdd_vocab_size, poi_vocab_size, embedding_dim, num_heads, num_layers, bert_out_dim):
#         super(DualLayerRecModel, self).__init__()
#
#         # 嵌入层
#         self.mdd_embedding = nn.Embedding(mdd_vocab_size, embedding_dim)        # 将每个城市 ID 转换成一个连续向量(后续通过索引来查表)
#         self.poi_embedding = nn.Embedding(poi_vocab_size, embedding_dim)        # 将每个城市 ID 转换成一个连续向量。
#         self.type_embedding = nn.Embedding(2, embedding_dim)     # 如果一个地点是 MDD 就查 type_embedding[0]，是 POI 就查 type_embedding[1]
#         self.domain_embedding = nn.Embedding(2, embedding_dim)  # domain (0: MFW, 1: FS)
#         self.bert_out_dim = bert_out_dim
#
#         self.norm = nn.LayerNorm(embedding_dim + self.bert_out_dim)
#         self.dropout = nn.Dropout(drop_prob)
#
#         # Transformer编码器（注意：PyTorch原生Transformer默认输入为 [seq_len, batch, d_model]）
#         self.transformer = nn.Transformer(
#             d_model=embedding_dim + self.bert_out_dim,  # 将地点ID嵌入（如 MDD/POI Embedding）和BERT得到的details向量拼接在一起
#             nhead=num_heads,    # 多头注意力机制的头数
#             num_encoder_layers=num_layers,    # 编码器层数
#             num_decoder_layers=num_layers,    # 解码器层数
#             dim_feedforward=embedding_dim * 4,
#             dropout=drop_prob,
#             activation='relu'
#         )
#
#         # 输出层(id和Bert得到的details的嵌入作为输入维度)
#         self.output_mdd = nn.Linear(embedding_dim + self.bert_out_dim, mdd_vocab_size)
#         self.output_poi = nn.Linear(embedding_dim + self.bert_out_dim, poi_vocab_size)
#         # 类型预测头
#         self.output_type = nn.Linear(embedding_dim + self.bert_out_dim, 2)  # 0: MDD, 1: POI
#     def forward(self, src, src_types, cls_features, domain_labels):
#         """
#             src: [batch_size, seq_len]         - 地点ID序列
#             src_types: [batch_size, seq_len]   - 地点类型（0=MDD，1=POI）
#         """
#
#         batch_size, seq_len = src.shape
#
#         # 构造地点嵌入
#         embeddings = torch.zeros(batch_size, seq_len, self.mdd_embedding.embedding_dim).to(src.device)
#         mdd_mask = (src_types == 0)
#         poi_mask = (src_types == 1)
#         embeddings[mdd_mask] = self.mdd_embedding(src[mdd_mask])
#         embeddings[poi_mask] = self.poi_embedding(src[poi_mask])
#
#         # 类型嵌入加和
#         type_embeds = self.type_embedding(src_types)
#         # print("domain_labels min/max:", domain_labels.min().item(), domain_labels.max().item())
#
#         domain_embeds = self.domain_embedding(domain_labels.unsqueeze(1).repeat(1, seq_len))  # [B, L] -> [B, L, D]
#         embeddings += type_embeds + domain_embeds
#         # embeddings += type_embeds  # [batch_size, seq_len, embed_dim]
#
#         combined_embeddings = torch.cat([embeddings, cls_features], dim=-1)  # [B, L, D + H]
#         combined_embeddings = self.norm(self.dropout(combined_embeddings))
#         # Transformer 输入要求：[seq_len, batch, dim]
#         transformer_input = combined_embeddings.permute(1, 0, 2)  # [L, B, D+H]
#
#         transformer_output = self.transformer(transformer_input, transformer_input)
#         transformer_output = transformer_output.permute(1, 0, 2)  # [B, L, D+H]
#
#         # 输出层
#         '''
#             # L=1，表示预测第一个位置后的一个地点的结果：
#             type_logits[0, 0] => [0.9, 0.1]      # 90% 可能是城市
#             mdd_logits[0, 0] => [0.1, 0.3, ..., 0.6]  # 对100个城市的logits分布
#             poi_logits[0, 0] => [0.01, ..., 0.05]     # 对1000个景点的logits分布
#         '''
#         mdd_logits = self.output_mdd(transformer_output)  # [B, L, mdd_vocab_size]
#         poi_logits = self.output_poi(transformer_output)  # [B, L, poi_vocab_size]
#         type_logits = self.output_type(transformer_output)  # [B, L, 2]
#         return mdd_logits, poi_logits, type_logits
