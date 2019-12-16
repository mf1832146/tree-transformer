import numpy as np
import torch
import torch.nn as nn
import torch.functional as F


def relative_mul(q, relative):
    """relative position dot product"""
    node_len, dim_per_head = relative.size(2), relative.size(3)
    relative_k = relative.transpose(2, 3).view(-1, dim_per_head, node_len)
    q = q.view(-1, dim_per_head).unsqueeze(1)
    return torch.bmm(q, relative_k).squeeze(1).view(-1, node_len, node_len)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()

        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None, relative_q=None, relative_v=None):
        """
                前向传播

                Args:
                    q : Queries shape [batch_size * num_heads, node_len, dim_per_head]
                    k : Keys shape [batch_size * num_heads, node_len, dim_per_head]
                    v: Values shape [batch_size * num_heads, node_len, dim_per_head]
                    scale: 缩放因子
                    attn_mask: 遮盖矩阵 shape [batch_size * num_heads, node_len, node_len]
                    relative_q: 相对位置编码 shape [batch_size * num_heads, node_len, node_len, dim_per_head]
                    relative_v: 相对位置编码 shape [batch_size * num_heads, node_len, node_len, dim_per_head]
                """
        attention = torch.bmm(q, k.transpose(1, 2))
        if relative_q:
            attention += relative_mul(q, relative_q)
        if scale:
            attention = attention * scale
        if attn_mask:
            # 给需要mask的地方设置一个负无穷（因为接下来要输入到softmax层，如果是0还是会有影响）
            attention = attention.mask_fill_(attn_mask, -np.int)
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        if relative_v:
            v += relative_mul(v, relative_v)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None, relative_q=None, relative_v=None):
        """
        muti-self attention
        :param key: shape [batch_size, num_heads, ]
        :param value:
        :param query:
        :param attn_mask:
        :param relative_q:
        :param relative_v:
        :return:
        """
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask, relative_q, relative_v)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # 残差连接
        output = self.layer_norm(output + residual)

        return output, attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_size):
        """
        :param d_model: 词向量维度，论文默认是512
        :param max_size: 深度不超过max_size, 任意节点的子节点个数不超过max_size
        """
        super(PositionalEncoding, self).__init__()

        # 根据论文公式，构造PE矩阵
        position_encoding = np.array([
          [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
          for pos in range(max_size)])
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        pad_row = torch.zeros([1, d_model])
        # 0为填充向量
        position_encoding = torch.cat((pad_row, position_encoding))
        self.position_encoding = nn.Embedding(max_size + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, inputs):
        """
        :param inputs: 位置矩阵, 即traverse中的brother_ids or
        parent_ids shape [batch_size, max_size]
        :return: 位置编码
        """
        # inputs扩一维度，变为 [batch_size, max_size, 1]
        inputs = inputs.unsqueeze(2)
        return self.position_encoding(inputs)


class RelativePositionEmbedding(nn.Module):
    def __init__(self, d_model, k, dropout=0.0):
        """
        生成相对位置信息编码
        :param d_model: 词向量维度
        :param k: 相对位置窗口大小
        :param dropout:
        """
        super(RelativePositionEmbedding, self).__init__()

        self.d_model = d_model
        self.k = k

        self.parent_emb = nn.Embedding((2*k+2, d_model * 2), padding_idx=0)
        self.brother_emb = nn.Embedding((2*k+2, d_model * 2), padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, relation_type):
        """
        :param inputs: 相对位置矩阵, 即traverse中的relative_parent_ids or
        relative_brother_ids shape [batch_size, max_size, max_size]
        :param relation_type: 'parent' means find relation between parent and child, 'brother' means find relation between brothers
        :return:
        """
        batch_size, max_size = inputs.size(0), inputs.size(1)
        inputs = inputs.unsqueeze(3)
        if relation_type == 'parent':
            position_emb = self.parent_emb(inputs)
        else:
            position_emb = self.brother_emb(inputs)
        position_emb = self.dropout(position_emb)
        position_emb = position_emb.view(batch_size, max_size, max_size, 2, self.d_model)
        k_emb, v_emb = [x.squeeze(3) for x in position_emb.split(1, dim=3)]
        return k_emb, v_emb


class PositionWiseFeedForward(nn.Module):
    """Position-wise feed forward network"""

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionWiseFeedForward, self).__init__()

        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        """
        Forward pass.
        :param x:  Input tensor, with shape of [batch_size, ]
        :return:
        """
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask_q=None, attn_mask_v=None):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask_q, attn_mask_v)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):
    def __init__(self,
                 voc_size,
                 max_size,
                 num_layers=6,
                 relative_pos=False,
                 k=5,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers * 2)]
        )
        self.seq_embedding = nn.Embedding(voc_size + 1, model_dim, padding_idx=0)
        self.relative_pos = relative_pos
        if self.relative_pos:
            self.dim_per_head = model_dim // num_heads
            self.relative_pos_embedding = RelativePositionEmbedding(self.dim_per_head, k, dropout)
        else:
            self.pos_embedding = PositionalEncoding(model_dim, max_size)

    def forward(self, inputs):
        seq, parent_matrix, brother_matrix, parent_ids, brother_ids, relative_parent_ids, relative_brother_ids = inputs
        """
        seq shape [batch_size, max_size],
        parent_matrix shape [batch_size, max_size, max_size]
        brother_matrix shape [batch_size, max_size, max_size]
        brother_ids shape [batch_size, max_size]
        parent_ids shape [batch_size, max_size]
        relative_parent_ids shape [batch_size, max_size, max_size]
        relative_brother_ids shape [batch_size, max_size, max_size]
        """
        # output = self.seq_embedding(seq)
        output = self.seq_embedding(seq)
        if self.relative_pos:
            parent_k_emb, parent_v_emb = self.relative_pos_embedding(relative_parent_ids, 'parent')
            brother_k_emb, brother_v_emb = self.relative_pos_embedding(relative_brother_ids, 'brother')

        attentions = []
        for i, encoder in enumerate(self.encoder_layers):
            if i % 2 == 0:
                # find brother relations
                output, attention = encoder(output, brother_matrix, brother_k_emb, brother_v_emb)
            else:
                # find parent relations
                output, attention = encoder(output, parent_matrix, parent_k_emb, parent_v_emb)

        attentions.append(attention)
        return output, attentions


# class Sentiment(nn.Module):
#     def __init__(self, encoder, mem_dim, num_classes, dropout=False):
#         self.encoder = encoder
#         self.mem_dim = mem_dim
#         self.num_classes = num_classes
#         self.dropout = dropout
#         # torch.manual_seed(456)
#         self.l1 = nn.Linear(self.mem_dim, self.num_classes)
#         self.logsoftmax = nn.LogSoftmax()
#         if self.cudaFlag:
#             self.l1 = self.l1.cuda()
#
#     def forward(self, inputs, training = False):
#         """
#         Sentiment module forward function
#         :param vec: (1, mem_dim)
#         :param training:
#         :return:
#         (1, number_of_class)
#         """
#         outputs, _ = self.encoder(inputs)
#         vec = outputs[:, 0]
#         if self.dropout:
#             out = self.logsoftmax(self.l1(F.dropout(vec, training=training)))
#         else:
#             out = self.logsoftmax(self.l1(vec))
#         return out


def padding_mask(seq_k, seq_q):
    len_q = seq_q.size(1)

    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)
    return pad_mask


def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                      diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class DecoderPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        """初始化。

        Args:
            d_model: 一个标量。模型的维度，论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        super(PositionalEncoding, self).__init__()

        # 根据论文给的公式，构造出PE矩阵
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))

        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len):
        """神经网络的前向传播。

        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """

        # 找出这一批序列的最大长度
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos = tensor(
            [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        return self.position_encoding(input_pos)


class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask=None, context_attn_mask=None):
        # self attention, all inputs are decoder inputs
        dec_output, self_attention = self.attention(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)

        # context attention
        dec_output, context_attention = self_attention(enc_outputs, enc_outputs, dec_output, context_attn_mask)

        # decoder's output, or cantext
        dec_output = self.feed_forward(dec_output)
        return dec_output, self_attention, context_attention


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])
        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = DecoderPositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len, enc_output, context_attn_mask=None):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)

        self_attention_padding_mask = padding_mask(inputs, inputs)
        seq_mask = sequence_mask(inputs)
        self_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0)

        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(
                output, enc_output, self_attn_mask, context_attn_mask)
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)

        return output, self_attentions, context_attentions


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 src_max_len,
                 tgt_vocab_size,
                 tgt_max_len,
                 relative_pos=True,
                 k=5,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.2):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, relative_pos,
                               k, model_dim, num_heads, ffn_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers, model_dim,
                               num_heads, ffn_dim, dropout)

        self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src_seq, src_len, tgt_seq, tgt_len):
        context_attn_mask = padding_mask(tgt_seq, src_seq)

        output, enc_self_attn = self.encoder(src_seq, src_len)
        output, dec_self_attn, ctx_attn = self.decoder(tgt_seq, tgt_len, output, context_attn_mask)

        output = self.linear(output)
        output = self.softmax(output)

        return output, enc_self_attn, dec_self_attn, ctx_attn
