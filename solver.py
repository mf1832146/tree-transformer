from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import TreeDataSet
from utils import make_save_dir
import os
import torch
import time
import numpy as np
from torch import nn
from utils import subsequent_mask, load_dict, log
from tree_transformer import EncoderDecoder, Encoder, Decoder, Generator, make_model
from bert_optimizer import BertAdam


class Solver:
    def __init__(self, args):
        self.args = args

        self.model_dir = make_save_dir(args.model_dir)
        self.model = self.make_model()

    def make_model(self):
        encoder = Encoder(voc_size=self.args.code_vocab_size,
                          max_size=self.args.code_max_len,
                          num_layers=self.args.num_layers,
                          relative_pos=self.args.relative_pos,
                          k=self.args.k,
                          model_dim=self.args.model_dim,
                          num_heads=self.args.num_heads,
                          ffn_dim=self.args.ffn_dim,
                          dropout=self.args.dropout)
        decoder = Decoder(vocab_size=self.args.comment_vocab_size,
                          max_seq_len=self.args.comment_max_len,
                          num_layers=self.args.num_layers,
                          model_dim=self.args.model_dim,
                          num_heads=self.args.num_heads,
                          ffn_dim=self.args.ffn_dim,
                          dropout=self.args.dropout)
        model = EncoderDecoder(encoder, decoder, Generator(self.args.model_dim, self.args.comment_vocab_size))

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

        #if torch.cuda.is_available:
        #    model = model.cuda()
        return model

    def train(self):
        if self.args.load:
            path = os.path.join(self.model_dir, 'model.pth')
            self.model.load_state_dict(torch.load(path)['state_dict'])

        tt = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                ttt = 1
                for s in param.data.size():
                    ttt *= s
                tt += ttt
        print('total param num:', tt)

        print('Loading training data...')
        train_data_set = TreeDataSet(self.args.train_data_set, self.args.code_max_len, skip=63000)
        test_data_set = TreeDataSet(self.args.test_data_set, self.args.code_max_len, skip=7860)

        train_loader = DataLoader(dataset=train_data_set, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_data_set, batch_size=1, shuffle=False)

        print('load training data finished')
        # optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
        optim = BertAdam(self.model.parameters(), lr=1e-4)
        criterion = LabelSmoothing(size=self.args.comment_vocab_size, padding_idx=0, smoothing=0.1)
        criterion = criterion.cuda()
        loss_compute = SimpleLossCompute(self.model.generator, criterion, optim)

        total_loss = []

        for step in range(self.args.num_step):
            self.model.train()

            start = time.time()
            step_loss = run_epoch(step, train_loader, self.model, loss_compute)
            elapsed = time.time() - start
            print('----------epoch: %d end, total loss= %f , train_time= %f Sec -------------' % (step, step_loss, elapsed))
            total_loss.append(step_loss)
            print('saving!!!!')

            model_name = 'model.pth'
            state = {'epoch': step, 'state_dict': self.model.state_dict()}
            torch.save(state, os.path.join(self.model_dir, model_name))
            # test
            self.model.eval()
            self.test(test_loader)

        print('training process end, total_loss is =', total_loss)

    def test(self, data_set_loader=None):
        if self.args.load:
            path = os.path.join(self.model_dir, 'model.pth')
            self.model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)['state_dict'])

        if data_set_loader is None:
            data_set = TreeDataSet(self.args.test_data_set, self.args.code_max_len, skip=7860)
            data_set_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=False)

        nl_i2w = load_dict(open('./data/nl_i2w.pkl', 'rb'))
        nl_w2i = load_dict(open('./data/nl_w2i.pkl', 'rb'))

        self.model.eval()
        log('_____贪心验证——end_______', './model/test.txt')
        for i, data_batch in enumerate(data_set_loader):
            code, par_matrix, bro_matrix, rel_par_ids, rel_bro_ids, comments = data_batch
            batch = Batch(code, par_matrix, bro_matrix, rel_par_ids, rel_bro_ids, None)
            log('Comment:' + ' '.join(nl_i2w[c.item()] for c in comments[0]), './model/test.txt')
            start_pos = nl_w2i['<s>']
            predicts = greedy_decode(self.model, batch, self.args.comment_max_len, start_pos)
            log('Predict:' + ' '.join(nl_i2w[c.item()] for c in predicts[0]), './model/test.txt')
        print('_____贪心验证——end_______')


def run_epoch(epoch, data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, data_batch in enumerate(data_iter):
        code, par_matrix, bro_matrix, rel_par_ids, rel_bro_ids, comments = data_batch
        batch = Batch(code, par_matrix, bro_matrix, rel_par_ids, rel_bro_ids, comments)
        batch = data_batch
        out, _, _, _ = model.forward(batch.code, batch.code_mask,
                                     batch.par_matrix, batch.bro_matrix,
                                     batch.re_par_ids, batch.re_bro_ids,
                                     batch.comments, batch.comment_mask)
        loss = loss_compute(out, batch.predicts, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens.item()
        tokens += batch.ntokens.item()
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch %d Step: %d Loss: %f Tokens per Sec: %f" %
                  (epoch, i, loss / batch.ntokens.item(), tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def greedy_decode(tree_transformer_model, batch, max_len, start_pos):
    memory, _ = tree_transformer_model.encode(batch.code,
                                              batch.par_matrix,
                                              batch.bro_matrix,
                                              batch.re_par_ids,
                                              batch.re_bro_ids)
    print(memory)
    ys = torch.ones(1, 1).fill_(start_pos).type_as(batch.code.data)
    for i in range(max_len - 1):
        #  memory, code_mask, comment, comment_mask
        out, _, _ = tree_transformer_model.decode(memory, batch.code_mask,
                                                  Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(batch.code.data)))
        if i == 0 :
            print('out', out)
        prob = tree_transformer_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(batch.code.data).fill_(next_word)], dim=1)
    return ys


class Batch:
    def __init__(self,
                 code,
                 par_matrix,
                 bro_matrix,
                 re_par_ids,
                 re_bro_ids,
                 comments=None,
                 pad=0):
        # 加载入gpu
        if torch.cuda.is_available():
            code = code.cuda()
            par_matrix = par_matrix.cuda()
            bro_matrix = bro_matrix.cuda()
            re_par_ids = re_par_ids.cuda()
            re_bro_ids = re_bro_ids.cuda()
            if comments is not None:
                comments = comments.cuda()

        self.code = code
        # code_mask用于解码时用
        self.code_mask = (code != pad).unsqueeze(-2)
        self.par_matrix = par_matrix
        self.bro_matrix = bro_matrix
        self.re_par_ids = re_par_ids
        self.re_bro_ids = re_bro_ids
        if comments is not None:
            self.comments = comments[:, :-1]
            self.predicts = comments[:, 1:]
            self.comment_mask = self.make_std_mask(self.comments, pad)
            # 训练时的有效预测个数
            self.ntokens = (self.predicts != pad).data.sum()

    @staticmethod
    def make_std_mask(comment, pad):
        comment_mask = (comment != pad).unsqueeze(-2)
        tgt_mask = comment_mask & Variable(
            subsequent_mask(comment.size(-1)).type_as(comment_mask.data))
        return tgt_mask


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
            # self.opt.zero_grad()
        return (loss * norm).item()


def data_gen(V, batch, nbatches):
    for z in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        parent_matrix = torch.zeros(batch, 10, 10).long()
        brother_matrix = torch.ones(batch, 10, 10).long()
        rel_par_matrix = torch.zeros(batch, 10, 10).long()
        for i in range(10):
            for j in range(10):
                rel_par_matrix[:, i, j] = int(max(-2, min(2, j-i)) + 2 + 1)
        rel_bro_matrix = torch.zeros(batch, 10, 10).long()

        code = Variable(data, requires_grad=False)
        par_matrix = Variable(parent_matrix, requires_grad=False)
        bro_matrix = Variable(brother_matrix, requires_grad=False)
        re_par_ids = Variable(rel_par_matrix, requires_grad=False)
        re_bro_ids = Variable(rel_bro_matrix, requires_grad=False)
        comments = Variable(data, requires_grad=False)

        yield Batch(code, par_matrix, bro_matrix, re_par_ids, re_bro_ids, comments)


if __name__ == '__main__':
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, 2)
    model_opt = NoamOpt(512, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch(1, data_gen(V, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))
        # model.eval()
        # print(run_epoch(1, data_gen(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None)))
        # print(greedy_decode(model, data_gen(V, 1, 20), max_len=10, start_pos=1))




