"""This is the file for main model."""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from settings import use_cuda, MAX_LENGTH, LAYER_DEPTH, TOCOPY


class Seq2Seq(object):
    def __init__(self, encoder, decoder, train_func, criterion, embedding_size, langs):
        self.encoder = encoder
        self.decoder = decoder
        self.train_func = train_func
        self.criterion = criterion
        self.embedding_size = embedding_size
        self.langs = langs

    def seq_train(self, rt, re, rm, summary):
        return self.train_func(rt, re, rm, summary,
                               self.encoder, self.decoder,
                               self.criterion, self.embedding_size, self.langs)

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()


class docEmbedding(nn.Module):
    def __init__(self, rt_size, re_size, rm_size, embedding_dim):
        super(docEmbedding, self).__init__()
        self.embedding1 = nn.Embedding(rt_size, embedding_dim)
        self.embedding2 = nn.Embedding(re_size, embedding_dim)
        self.embedding3 = nn.Embedding(rm_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim * 3, embedding_dim)

    def forward(self, rt, re, rm):
        emb_rt = self.embedding1(rt)
        emb_re = self.embedding2(re)
        emb_rm = self.embedding3(rm)

        emb_all = torch.cat([emb_rt, emb_re, emb_rm], dim=len(rt.size()))
        output = self.linear(emb_all)
        return output

    def init_weights(self):
        initrange = 0.1
        lin_layers = [self.linear]
        em_layer = [self.embedding1, self.embedding2, self.embedding3]

        for layer in lin_layers + em_layer:
            layer.weight.data.uniform_(-initrange, initrange)
            if layer in lin_layers:
                layer.bias.data.fill_(0)


class HierarchicalRNN(nn.Module):
    def __init__(self, hidden_size, local_embed, n_layers=LAYER_DEPTH):
        super(HierarchicalRNN, self).__init__()
        self.LocalEncoder = EncoderRNN(hidden_size, local_embed,
                                       n_layers=n_layers, level='local')
        self.GlobalEncoder = EncoderRNN(hidden_size, None,
                                        n_layers=n_layers, level='global')


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_layer, n_layers=LAYER_DEPTH, level='local'):
        super(EncoderRNN, self).__init__()
        self.level = level
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        if self.level == 'local':
            self.embedding = embedding_layer
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=self.n_layers)

    def forward(self, inputs, hidden):
        if self.level == 'local':
            embedded = self.embedding(inputs['rt'], inputs['re'], inputs['rm'])
            inp = embedded.permute(1, 0, 2)
            seq_len = embedded.size(1)
            batch_size = embedded.size(0)
            embed_dim = embedded.size(2)
            outputs = Variable(torch.zeros(seq_len, batch_size, embed_dim))
            outputs = outputs.cuda() if use_cuda else outputs

            for ei in range(seq_len):
                if ei > 0 and ei % 32 == 0:
                    hidden = self.initHidden(batch_size)
                seq_i = inp[ei, :, :].unsqueeze(0)
                output, hidden = self.gru(seq_i, hidden)
                outputs[ei, :, :] = output[0, :, :]
        else:
            outputs, hidden = self.gru(inputs['local_hidden_states'], hidden)
        return outputs, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size), requires_grad=False)

        if use_cuda:
            return result.cuda()
        else:
            return result


class PGenLayer(nn.Module):
    def __init__(self, emb_dim, hidden_size, enc_dim):
        super(PGenLayer, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.enc_dim = enc_dim
        self.lin = nn.Linear(self.emb_dim + self.hidden_size + self.enc_dim, 1)

    def forward(self, emb, hid, enc):
        input = torch.cat((emb, hid, enc), 1)
        return F.sigmoid(self.lin(input))


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=LAYER_DEPTH, 
                 dropout_p=0.1, copy=TOCOPY):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.copy = copy

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = Attn(hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size * 2, self.output_size)
        if self.copy:
            self.pgen = PGenLayer(self.hidden_size, self.hidden_size, self.hidden_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)

        attn_weights = self.attn(hidden[-1, :, :], encoder_outputs)
        context = torch.bmm(attn_weights, encoder_outputs)

        context = context.squeeze(1)
        output = torch.cat((embedded, context), dim=1)
        output = output.unsqueeze(0)
        output, nh = self.gru(output, hidden)

        output = output.squeeze(0)

        if self.copy:
            pgen = self.pgen(embedded, output, context)
            output = F.log_softmax(self.out(torch.cat((output, context), 1)), dim=1) + pgen.log()
        else:
            pgen = 0
            output = F.log_softmax(self.out(torch.cat((output, context), 1)), dim=1)

        return output, nh, context, attn_weights, pgen

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size), requires_grad=False)
        if use_cuda:
            return result.cuda()
        else:
            return result


class HierarchicalDecoder(nn.Module):
    def __init__(self, hidden_size, output_size,
                 n_layers=LAYER_DEPTH, copy=TOCOPY):

        super(HierarchicalDecoder, self).__init__()
        self.global_decoder = GlobalAttnDecoderRNN(hidden_size, n_layers=n_layers)
        self.local_decoder = LocalAttnDecoderRNN(hidden_size, output_size,
                                                 n_layers=n_layers, copy=copy)


class GlobalAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, n_layers=LAYER_DEPTH, dropout_p=0.1):
        super(GlobalAttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.attn = Attn(hidden_size)

        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)

    def forward(self, input, hidden, encoder_outputs):

        attn_weights = self.attn(hidden[-1, :, :], encoder_outputs)
        context = torch.bmm(attn_weights, encoder_outputs)
        output = torch.cat((input, context.squeeze(1)), dim=1)

        output = output.unsqueeze(0)
        output, nh = self.gru(output, hidden)

        return output, nh, context, attn_weights

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size), requires_grad=False)
        if use_cuda:
            return result.cuda()
        else:
            return result


class LocalAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=MAX_LENGTH,
                 n_layers=LAYER_DEPTH, dropout_p=0.1, copy=TOCOPY):
        super(LocalAttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.copy = copy
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = Attn(hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)

        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)

        self.out = nn.Linear(self.hidden_size * 2, self.output_size)
        if self.copy:
            self.pgen = PGenLayer(self.hidden_size, self.hidden_size, self.hidden_size)

    def forward(self, input, hidden, block_attn_weights, encoder_outputs, blocks):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        batch_size_blk_size, seq_len, hidden_size = encoder_outputs.size()
        batch_size = batch_size_blk_size // len(blocks)

        hid = hidden[-1, :, :]
        hid = hid.repeat(len(blocks), 1)

        attn_weights = self.attn(hid, encoder_outputs)

        block_context = torch.bmm(attn_weights, encoder_outputs)  # (batch * blk, 1, hid)
        block_context = block_context.view(batch_size, len(blocks), hidden_size)

        context = torch.bmm(block_attn_weights, block_context)

        context = context.squeeze(1)

        output = torch.cat((embedded, context), dim=1)

        output = output.unsqueeze(0)
        output, nh = self.gru(output, hidden)

        output = output.squeeze(0)

        if self.copy:
            pgen = self.pgen(embedded, output, context)
            output = F.log_softmax(self.out(torch.cat((output, context), 1)), dim=1) + pgen.log()
        else:
            pgen = Variable(torch.zeros(1, 1)).cuda() if use_cuda else Variable(torch.zeros(1, 1))
            output = F.log_softmax(self.out(torch.cat((output, context), 1)), dim=1)

        return output, nh, context, attn_weights, pgen

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size), requires_grad=False)
        if use_cuda:
            return result.cuda()
        else:
            return result


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden, encoder_outputs):
        batch_size, seq_len, hidden_size = encoder_outputs.size()
        hidden = hidden.unsqueeze(1)  # (batch_size, 1, hidden_size)
        hiddens = hidden.repeat(1, seq_len, 1)
        attn_energies = self.score(hiddens, encoder_outputs)

        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        energy = self.attn(encoder_outputs)

        hidden = hidden.unsqueeze(2)  # (batch, seq, 1, d)
        energy = energy.unsqueeze(3)  # (batch, seq, d, 1)

        energy = torch.matmul(hidden, energy)  # (batch, seq, 1, 1)

        return energy.squeeze(3).squeeze(2)

