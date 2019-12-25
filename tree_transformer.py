import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import gelu
from torch.autograd import Variable


# def relative_mul(q, relative):
#     """relative position dot product"""
#     batch_size, node_len, dim_per_head = relative.size(0), relative.size(2), relative.size(3)
#     relative_k = relative.transpose(2, 3).view(-1, dim_per_head, node_len)
#     q = q.view(batch_size, -1, node_len, dim_per_head)
#     q = q.transpose(1, 2)
#     q = q.contiguous().view(batch_size * node_len, -1, dim_per_head)
#     return torch.bmm(q, relative_k).view(batch_size, node_len, -1, node_len).transpose(2, 3).contiguous().view(-1, node_len, node_len)

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
        if relative_q is not None:
            attention += relative_mul(q, relative_q)
        if scale is not None:
            attention = attention * scale
        if attn_mask is not None:
            # 给需要mask的地方设置一个负无穷（因为接下来要输入到softmax层，如果是0还是会有影响）
            attention = attention.masked_fill_(mask=attn_mask.bool(), value=-1e9)
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)
        if relative_v is not None:
            node_len, dim_per_head = relative_v.size(2), relative_v.size(3)
            att_v = attention.view(-1, node_len).unsqueeze(1)
            relative_v = relative_v.view(-1, node_len, dim_per_head)
            context_v = torch.bmm(att_v, relative_v).squeeze(1)
            context_v = context_v.view(-1, node_len, dim_per_head)
            context += context_v

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

        if attn_mask is not None:
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


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_size):
#         """
#         :param d_model: 词向量维度，论文默认是512
#         :param max_size: 深度不超过max_size, 任意节点的子节点个数不超过max_size
#         """
#         super(PositionalEncoding, self).__init__()
#
#         # 根据论文公式，构造PE矩阵
#         position_encoding = np.array([
#           [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
#           for pos in range(max_size)])
#         # 偶数列使用sin，奇数列使用cos
#         position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
#         position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
#
#         pad_row = torch.zeros([1, d_model])
#         # 0为填充向量
#         position_encoding = torch.cat((pad_row, position_encoding))
#         self.position_encoding = nn.Embedding(max_size + 1, d_model)
#         self.position_encoding.weight = nn.Parameter(position_encoding,
#                                                      requires_grad=False)
#
#     def forward(self, inputs):
#         """
#         :param inputs: 位置矩阵, 即traverse中的brother_ids or
#         parent_ids shape [batch_size, max_size]
#         :return: 位置编码
#         """
#         # inputs扩一维度，变为 [batch_size, max_size, 1]
#         inputs = inputs.unsqueeze(2)
#         return self.position_encoding(inputs)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return x


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

        self.parent_emb = nn.Embedding(2*k+2, d_model * 2, padding_idx=0)
        self.brother_emb = nn.Embedding(2*k+2, d_model * 2, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, relation_type, num_heads):
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

        k_emb = k_emb.repeat(1, 1, 1, num_heads)
        v_emb = v_emb.repeat(1, 1, 1, num_heads)

        k_emb = k_emb.view(-1, max_size, max_size, self.d_model)
        v_emb = v_emb.view(-1, max_size, max_size, self.d_model)

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
        output = self.w2(gelu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None, relative_q=None, relative_v=None):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask, relative_q, relative_v)

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
        self.num_heads = num_heads
        self.seq_embedding = nn.Embedding(voc_size, model_dim, padding_idx=0)
        self.relative_pos = relative_pos
        if self.relative_pos:
            self.dim_per_head = model_dim // num_heads
            self.relative_pos_embedding = RelativePositionEmbedding(self.dim_per_head, k, dropout)

    def forward(self, seq, parent_matrix, brother_matrix, parent_ids, brother_ids, relative_parent_ids, relative_brother_ids):
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
            parent_k_emb, parent_v_emb = self.relative_pos_embedding(relative_parent_ids, 'parent', self.num_heads)
            brother_k_emb, brother_v_emb = self.relative_pos_embedding(relative_brother_ids, 'brother', self.num_heads)

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


# class DecoderPositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_seq_len):
#         """初始化。
#
#         Args:
#             d_model: 一个标量。模型的维度，论文默认是512
#             max_seq_len: 一个标量。文本序列的最大长度
#         """
#         super(DecoderPositionalEncoding, self).__init__()
#
#         # 根据论文给的公式，构造出PE矩阵
#         position_encoding = np.array([
#             [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
#             for pos in range(max_seq_len)])
#         # 偶数列使用sin，奇数列使用cos
#         position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
#         position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
#
#         # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
#         # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
#         # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
#         # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
#         pad_row = torch.zeros([1, d_model])
#         position_encoding = torch.cat((pad_row, position_encoding))
#
#         # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
#         # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
#         self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
#         self.position_encoding.weight = nn.Parameter(position_encoding,
#                                                      requires_grad=False)
#
#     def forward(self, input_len):
#         """神经网络的前向传播。
#
#         Args:
#           input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。
#
#         Returns:
#           返回这一批序列的位置编码，进行了对齐。
#         """
#
#         # 找出这一批序列的最大长度
#         max_len = torch.max(input_len)
#         tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
#         # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
#         # 这里range从1开始也是因为要避开PAD(0)的位置
#         input_pos = tensor(
#             [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
#         return self.position_encoding(input_pos)


class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(model_dim, num_heads, dropout)
        self.src_attn = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, comments, memory, code_mask=None, comment_mask=None):
        # self attention, all inputs are decoder inputs
        comments, comment_attention = self.self_attn(comments, comments, comments, comment_mask)

        # context attention
        comments, code_attention = self.src_attn(memory, memory, comments, code_mask)

        # decoder's output, or context
        comments = self.feed_forward(comments)
        return comments, comment_attention, code_attention


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
        self.seq_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, comments, memory, code_mask, comment_mask=None):
        comments = self.seq_embedding(comments)
        comments = self.pos_embedding(comments)
        comment_attn = []
        code_attn = []
        for layer in self.decoder_layers:
            comments, comment_attention, code_attention = layer(comments, memory, code_mask, comment_mask)
            comment_attn.append(comment_attention)
            code_attn.append(code_attention)
        return comments, comment_attn, code_attn

# class Transformer(nn.Module):
#     def __init__(self,
#                  src_vocab_size,
#                  src_max_len,
#                  tgt_vocab_size,
#                  tgt_max_len,
#                  relative_pos=True,
#                  k=5,
#                  num_layers=6,
#                  model_dim=512,
#                  num_heads=8,
#                  ffn_dim=2048,
#                  dropout=0.2):
#         super(Transformer, self).__init__()
#
#         self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, relative_pos,
#                                k, model_dim, num_heads, ffn_dim, dropout)
#         self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers, model_dim,
#                                num_heads, ffn_dim, dropout)
#
#         self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
#         self.softmax = nn.Softmax(dim=2)
#
#     def forward(self, src_seq, src_len, tgt_seq, tgt_len):
#         context_attn_mask = padding_mask(tgt_seq, src_seq)
#
#         output, enc_self_attn = self.encoder(src_seq, src_len)
#         output, dec_self_attn, ctx_attn = self.decoder(tgt_seq, tgt_len, output, context_attn_mask)
#
#         output = self.linear(output)
#         output = self.softmax(output)
#
#         return output, enc_self_attn, dec_self_attn, ctx_attn


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, code_seq, code_mask, par_matrix, bro_matrix, re_par_ids, re_bro_ids, comment, comment_mask):
        """
        :param code_seq:      代码序列
        :param code_mask:     代码序列遮盖，用于解码阶段
        :param par_matrix:    父子关系遮盖矩阵
        :param bro_matrix:    兄弟关系遮盖矩阵
        :param re_par_ids:    相对父子位置矩阵
        :param re_bro_ids:    相对兄弟位置矩阵
        :param comment:       评论序列
        :param comment_mask:  评论遮盖矩阵
        """
        encoder_output, encoder_attn = self.encode(code_seq, par_matrix, bro_matrix, re_par_ids, re_bro_ids)
        decoder_output, comment_attn, code_attn = self.decode(encoder_output, code_mask, comment, comment_mask)
        return decoder_output, comment_attn, code_attn, encoder_attn

    def encode(self, code_seq, par_matrix, bro_matrix, re_par_ids, re_bro_ids):
        return self.encoder(code_seq, par_matrix, bro_matrix, None, None, re_par_ids, re_bro_ids)

    def decode(self, memory, code_mask, comment, comment_mask):
        return self.decoder(comment, memory, code_mask, comment_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def make_model(code_vocab_size, comment_vocab_size,
               max_code_len=100, max_comment_len=100,
               relative_pos=True, k=2,
               N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    encoder = Encoder(voc_size=code_vocab_size,
                      max_size=max_code_len,
                      num_layers=N,
                      relative_pos=relative_pos,
                      k=k,
                      model_dim=d_model,
                      num_heads=h,
                      ffn_dim=d_ff,
                      dropout=dropout)
    decoder = Decoder(vocab_size=comment_vocab_size,
                      max_seq_len=max_comment_len,
                      num_layers=N,
                      model_dim=d_model,
                      num_heads=h,
                      ffn_dim=d_ff,
                      dropout=dropout)
    model = EncoderDecoder(encoder, decoder, Generator(d_model, comment_vocab_size))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model


if __name__ == '__main__':
    tmp_model = make_model(10, 10, 2)
    None