import torch
import torch.nn as nn
from transformers import BertModel,AutoModel

class DualLayerRecModel(nn.Module):
    def __init__(self, mdd_vocab_size, poi_vocab_size, embedding_dim, num_heads, num_layers,bert_model_name='distilbert-base-multilingual-cased'):
                 # bert_model_name='bert-base-multilingual-cased'):
        super(DualLayerRecModel, self).__init__()

        # 嵌入层
        self.mdd_embedding = nn.Embedding(mdd_vocab_size, embedding_dim)
        self.poi_embedding = nn.Embedding(poi_vocab_size, embedding_dim)
        self.type_embedding = nn.Embedding(2, embedding_dim)

        # BERT模型处理details
        # self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert_out_dim = self.bert.config.hidden_size  # 通常是768

        # Transformer编码器（注意：PyTorch原生Transformer默认输入为 [seq_len, batch, d_model]）
        self.transformer = nn.Transformer(
            d_model=embedding_dim + self.bert_out_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation='relu'
        )

        # 输出层
        self.output_mdd = nn.Linear(embedding_dim + self.bert_out_dim, mdd_vocab_size)
        self.output_poi = nn.Linear(embedding_dim + self.bert_out_dim, poi_vocab_size)

    def forward(self, src, src_types, details_input_ids, details_attention_mask):
        """
        src: [batch_size, seq_len]         - 地点ID序列
        src_types: [batch_size, seq_len]   - 地点类型（0=MDD，1=POI）
        details_input_ids: [batch_size, seq_len, 30]         - 文本输入ID
        details_attention_mask: [batch_size, seq_len, 30]    - 文本注意力Mask
        """

        batch_size, seq_len = src.shape

        # 构造地点嵌入
        embeddings = torch.zeros(batch_size, seq_len, self.mdd_embedding.embedding_dim).to(src.device)
        mdd_mask = (src_types == 0)
        poi_mask = (src_types == 1)
        embeddings[mdd_mask] = self.mdd_embedding(src[mdd_mask])
        embeddings[poi_mask] = self.poi_embedding(src[poi_mask])

        # 类型嵌入加和
        type_embeds = self.type_embedding(src_types)
        embeddings += type_embeds  # [batch_size, seq_len, embed_dim]

        # BERT 文本特征提取
        bert_len = details_input_ids.shape[2]  # 一般为30
        flat_input_ids = details_input_ids.view(-1, bert_len)  # [B*L, 30]
        flat_attention_mask = details_attention_mask.view(-1, bert_len)

        bert_output = self.bert(input_ids=flat_input_ids, attention_mask=flat_attention_mask).last_hidden_state
        cls_embeddings = bert_output[:, 0, :]  # [B*L, hidden]
        cls_embeddings = cls_embeddings.view(batch_size, seq_len, -1)  # [B, L, hidden]

        # 合并地点嵌入和文本嵌入
        combined_embeddings = torch.cat([embeddings, cls_embeddings], dim=-1)  # [B, L, D+H]

        # Transformer 输入要求：[seq_len, batch, dim]
        transformer_input = combined_embeddings.permute(1, 0, 2)  # [L, B, D+H]
        transformer_output = self.transformer(transformer_input, transformer_input)
        transformer_output = transformer_output.permute(1, 0, 2)  # [B, L, D+H]

        # 输出层
        mdd_logits = self.output_mdd(transformer_output)  # [B, L, mdd_vocab_size]
        poi_logits = self.output_poi(transformer_output)  # [B, L, poi_vocab_size]

        return mdd_logits, poi_logits
