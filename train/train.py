import time
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from preprocessing import data_iter
from dataprepare import loaddata, data2index
from model import docEmbedding, Seq2Seq
from model import HierarchicalRNN
from model import EncoderRNN
from model import AttnDecoderRNN, HierarchicalDecoder
from util import gettime, load_model, show_triplets
from util import PriorityQueue

from settings import file_loc, use_cuda
from settings import EMBEDDING_SIZE, LR, EPOCH_TIME, BATCH_SIZE, GRAD_CLIP
from settings import MAX_SENTENCES, ENCODER_STYLE, DECODER_STYLE, TOCOPY
from settings import GET_LOSS, SAVE_MODEL, OUTPUT_FILE, COPY_PLAYER, MAX_LENGTH
from settings import LAYER_DEPTH, PRETRAIN, MAX_TRAIN_NUM, iterNum

import numpy as np

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2
EOB_TOKEN = 4
BLK_TOKEN = 5


def get_batch(batch):
    batch_data = []
    batch_idx_data = [[], [], [], []]
    for d in batch:
        idx_data = [[], [], []]  # for each triplet
        batch_data.append([d.triplets, d.summary])  # keep the original data/ not indexed version
        for triplets in d.idx_data[0]:
            for idt, t in enumerate(triplets):
                idx_data[idt].append(t)

        for idb, b in enumerate(idx_data):
            batch_idx_data[idb].append(b)

        batch_idx_data[3].append(d.idx_data[1])

    return batch_data, batch_idx_data


def find_max_block_numbers(batch_length, langs, rm):
    blocks_lens = [[0] for i in range(batch_length)]
    BLOCK_NUMBERS = np.ones(batch_length)
    for bi in range(batch_length):
        for ei in range(len(rm[bi, :])):
            if langs['rm'].index2word[int(rm[bi, ei].data[0])] == '<EOB>':
                blocks_lens[bi].append(ei)
                BLOCK_NUMBERS[bi] += 1
    return int(np.max(BLOCK_NUMBERS)), blocks_lens


def initGlobalEncoderInput(MAX_BLOCK, batch_length, input_length, embedding_size,
                           local_outputs, BLOCK_JUMPS=32):
    global_input = Variable(torch.zeros(MAX_BLOCK, batch_length,
                                        embedding_size))
    global_input = global_input.cuda() if use_cuda else global_input
    for ei in range(1, input_length + 1):
        if ei % BLOCK_JUMPS == 0:
            block_idx = int(ei / (BLOCK_JUMPS + 1))
            global_input[block_idx, :, :] = local_outputs[ei - 1, :, :]
    return global_input


def sequenceloss(rt, re, rm, summary, model):
    return model.seq_train(rt, re, rm, summary)


def Hierarchical_seq_train(rt, re, rm, summary, encoder, decoder,
                           criterion, embedding_size, langs):
    batch_length = rt.size()[0]
    input_length = rt.size()[1]
    target_length = summary.size()[1]

    MAX_BLOCK, blocks_lens = find_max_block_numbers(batch_length, langs, rm)

    inputs = {"rt": rt, "re": re, "rm": rm}

    LocalEncoder = encoder.LocalEncoder
    GlobalEncoder = encoder.GlobalEncoder

    loss = 0

    init_local_hidden = LocalEncoder.initHidden(batch_length)
    init_global_hidden = GlobalEncoder.initHidden(batch_length)
    local_encoder_outputs, local_hidden = LocalEncoder(inputs, init_local_hidden)
    global_input = initGlobalEncoderInput(MAX_BLOCK, batch_length, input_length,
                                          embedding_size, local_encoder_outputs)
    global_encoder_outputs, global_hidden = GlobalEncoder({"local_hidden_states":
                                                          global_input}, init_global_hidden)

    local_encoder_outputs = local_encoder_outputs.permute(1, 0, 2)
    global_encoder_outputs = global_encoder_outputs.permute(1, 0, 2)

    global_decoder = decoder.global_decoder
    local_decoder = decoder.local_decoder

    blocks_len = blocks_lens[0]

    g_input = global_decoder.initHidden(batch_length).permute(1, 2, 0)[:, :, -1]
    gnh = global_hidden

    l_input = Variable(torch.LongTensor(batch_length).zero_(), requires_grad=False)
    l_input = l_input.cuda() if use_cuda else l_input

    lnh = local_decoder.initHidden(batch_length)

    local_encoder_outputs = local_encoder_outputs.contiguous().view(batch_length * len(blocks_len),
                                                                    input_length // len(blocks_len),
                                                                    embedding_size)
    for di in range(target_length):
        if di == 0 or l_input[0].data[0] == BLK_TOKEN:
            g_output, gnh, g_context, g_attn_weights = global_decoder(
                g_input, gnh, global_encoder_outputs)

            lnh = gnh

        l_output, lnh, l_context, l_attn_weights, pgen = local_decoder(
            l_input, lnh, g_attn_weights, local_encoder_outputs, blocks_len)

        if local_decoder.copy:
            l_attn_weights = l_attn_weights.squeeze(1)
            bg_attn_weights = g_attn_weights.view(batch_length * len(blocks_len), -1)

            combine_attn_weights = l_attn_weights * bg_attn_weights

            combine_attn_weights = combine_attn_weights.view(batch_length, -1)

            prob = Variable(torch.zeros(l_output.shape), requires_grad=False)
            prob = prob.cuda() if use_cuda else prob

            prob = prob.scatter_add(1, rm, combine_attn_weights)

            l_output_new = (l_output.exp() + (1 - pgen) * prob).log()
        else:
            l_output_new = l_output

        loss += criterion(l_output_new, summary[:, di])
        g_input = lnh[-1, :, :]
        l_input = summary[:, di]  # Supervised

    return loss


def add_sentence_paddings(summarizes):
    def len_block(summary):
        return summary.count(BLK_TOKEN)

    max_blocks_length = max(list(map(len_block, summarizes)))

    for i in range(len(summarizes)):
        summarizes[i] += [BLK_TOKEN for j in range(max_blocks_length - len_block(summarizes[i]))]

    def to_matrix(summary):
        mat = [[] for i in range(len_block(summary) + 1)]
        idt = 0
        for word in summary:
            if word == BLK_TOKEN:
                idt += 1
            else:
                mat[idt].append(word)
        return mat

    for i in range(len(summarizes)):
        summarizes[i] = to_matrix(summarizes[i])

    def len_sentence(matrix):
        return max(list(map(len, matrix)))

    max_sentence_length = max([len_sentence(s) for s in summarizes])
    for i in range(len(summarizes)):
        for j in range(len(summarizes[i])):
            summarizes[i][j] += [PAD_TOKEN for k in range(max_sentence_length - len(summarizes[i][j]))]
            summarizes[i][j] += [BLK_TOKEN]

    def to_list(matrix):
        return [j for i in matrix for j in i]

    for i in range(len(summarizes)):
        summarizes[i] = to_list(summarizes[i])

    return summarizes


def addpaddings(tokens):
    max_length = len(max(tokens, key=len))
    for i in range(len(tokens)):
        tokens[i] += [PAD_TOKEN for i in range(max_length - len(tokens[i]))]
    return tokens


def train(train_set, langs, embedding_size=EMBEDDING_SIZE, learning_rate=LR,
          batch_size=BATCH_SIZE, get_loss=GET_LOSS, grad_clip=GRAD_CLIP,
          encoder_style=ENCODER_STYLE, decoder_style=DECODER_STYLE,
          to_copy=TOCOPY, epoch_time=EPOCH_TIME, layer_depth=LAYER_DEPTH,
          max_length=MAX_LENGTH, max_sentence=MAX_SENTENCES,
          save_model=SAVE_MODEL, output_file=OUTPUT_FILE,
          iter_num=iterNum, pretrain=PRETRAIN):

    start = time.time()

    emb = docEmbedding(langs['rt'].n_words, langs['re'].n_words,
                       langs['rm'].n_words, embedding_size)
    emb.init_weights()

    encoder_args = {"hidden_size": embedding_size, "local_embed": emb,
                        "n_layers": layer_depth}
    encoder = HierarchicalRNN(**encoder_args)

    if decoder_style == 'HierarchicalRNN':
        decoder = HierarchicalDecoder(embedding_size, langs['summary'].n_words,
                                      n_layers=layer_depth, copy=to_copy)
        train_func = Hierarchical_seq_train
    else:
        decoder = AttnDecoderRNN(embedding_size, langs['summary'].n_words,
                                 n_layers=layer_depth, copy=to_copy)
        train_func = Plain_seq_train

    if use_cuda:
        emb.cuda()
        encoder.cuda()
        decoder.cuda()

    loss_optimizer = optim.Adagrad(list(encoder.parameters()) + list(decoder.parameters()),
                                   lr=learning_rate, lr_decay=0, weight_decay=0)

    use_model = None
    if pretrain is not None and iter_num is not None:
        use_model = ['./models/' + pretrain + '_' + s + '_' + str(iter_num)
                     for s in ['encoder', 'decoder', 'optim']]

    if use_model is not None:
        encoder = load_model(encoder, use_model[0])
        decoder = load_model(decoder, use_model[1])
        loss_optimizer.load_state_dict(torch.load(use_model[2]))
        print("Load Pretrain Model {}".format(use_model))
    else:
        print("Not use Pretrain Model")

    criterion = nn.NLLLoss()

    model = Seq2Seq(encoder, decoder, train_func, criterion, embedding_size, langs)

    total_loss = 0
    iteration = 0
    for epo in range(1, epoch_time + 1):
        print("Epoch #%d" % (epo))

        train_iter = data_iter(train_set, batch_size=batch_size)
        for dt in train_iter:
            iteration += 1
            data, idx_data = get_batch(dt)
            rt, re, rm, summary = idx_data

            rt = addpaddings(rt)
            re = addpaddings(re)
            rm = addpaddings(rm)

            if decoder_style == 'HierarchicalRNN' and batch_size != 1:
                summary = add_sentence_paddings(summary)
            else:
                summary = addpaddings(summary)

            rt = Variable(torch.LongTensor(rt), requires_grad=False)
            re = Variable(torch.LongTensor(re), requires_grad=False)
            rm = Variable(torch.LongTensor(rm), requires_grad=False)

            summary = Variable(torch.LongTensor(summary), requires_grad=False)

            if use_cuda:
                rt, re, rm, summary = rt.cuda(), re.cuda(), rm.cuda(), summary.cuda()

            loss_optimizer.zero_grad()
            model.train()

            loss = sequenceloss(rt, re, rm, summary, model)

            loss.backward()
            torch.nn.utils.clip_grad_norm(list(model.encoder.parameters()) +
                                          list(model.decoder.parameters()),
                                          grad_clip)
            loss_optimizer.step()

            target_length = summary.size()[1]
            if float(torch.__version__[:3]) > 0.3:
                total_loss += loss.item() / target_length
            else:
                total_loss += loss.data[0] / target_length

            if iteration % get_loss == 0:
                print("Time {}, iter {}, Seq_len:{}, avg loss = {:.4f}".format(
                    gettime(start), iteration, target_length, total_loss / get_loss))
                total_loss = 0

        if epo % save_model == 0:
            torch.save(encoder.state_dict(),
                       "models/{}_encoder_{}".format(output_file, iteration))
            torch.save(decoder.state_dict(),
                       "models/{}_decoder_{}".format(output_file, iteration))
            torch.save(loss_optimizer.state_dict(),
                       "models/{}_optim_{}".format(output_file, iteration))
            print("Save the model at iter {}".format(iteration))

    return model.encoder, model.decoder


def hierarchical_predictwords(rt, re, rm, summary, encoder, decoder, lang,
                              embedding_size, encoder_style, beam_size):
    batch_length = rt.size()[0]
    input_length = rt.size()[1]
    target_length = 1000

    MAX_BLOCK, blocks_lens = find_max_block_numbers(batch_length, lang, rm)
    BLOCK_JUMPS = 32

    LocalEncoder = encoder.LocalEncoder
    GlobalEncoder = encoder.GlobalEncoder

    local_encoder_outputs = Variable(torch.zeros(batch_length, input_length, embedding_size))
    local_encoder_outputs = local_encoder_outputs.cuda() if use_cuda else local_encoder_outputs
    global_encoder_outputs = Variable(torch.zeros(batch_length, MAX_BLOCK, embedding_size))
    global_encoder_outputs = global_encoder_outputs.cuda() if use_cuda else global_encoder_outputs

    if encoder_style == 'BiLSTM':
        init_hidden = encoder.initHidden(batch_length)
        encoder_hidden, encoder_hiddens = encoder(rt, re, rm, init_hidden)

        for ei in range(input_length):
            encoder_outputs[:, ei] = encoder_hiddens[:, ei]

    else:
        init_local_hidden = LocalEncoder.initHidden(batch_length)
        local_out, local_hidden = LocalEncoder({"rt": rt, "re": re, "rm": rm},
                                               init_local_hidden)
        
        global_input = Variable(torch.zeros(MAX_BLOCK, batch_length,
                                            embedding_size))
        global_input = global_input.cuda() if use_cuda else global_input
        for ei in range(input_length):
            if ei % BLOCK_JUMPS == 0:
                global_input[int(ei / (BLOCK_JUMPS + 1)), :, :] = local_out[ei, :, :]

        init_global_hidden = GlobalEncoder.initHidden(batch_length)
        global_out, global_hidden = GlobalEncoder({"local_hidden_states":
                                                  global_input}, init_global_hidden)

        local_encoder_outputs = local_out.permute(1, 0, 2)
        global_encoder_outputs = global_out.permute(1, 0, 2)

    global_decoder = decoder.global_decoder
    local_decoder = decoder.local_decoder

    blocks_len = blocks_lens[0]

    gnh = global_decoder.initHidden(batch_length)
    lnh = local_decoder.initHidden(batch_length)

    g_input = global_encoder_outputs[:, -1]
    l_input = Variable(torch.LongTensor(batch_length).zero_(), requires_grad=False)
    l_input = l_input.cuda() if use_cuda else l_input

    decoder_attentions = torch.zeros(target_length, input_length)

    beams = [[0, [SOS_TOKEN], encoder_hidden, decoder_attentions]]

    for di in range(target_length):

        q = PriorityQueue()
        for beam in beams:

            prob, route, decoder_hidden, atten = beam
            destination = len(route) - 1

            decoder_input = route[-1]

            if decoder_input == EOS_TOKEN:
                q.push(beam, prob)
                continue

            decoder_input = Variable(torch.LongTensor([decoder_input]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            atten[destination, :decoder_attention.shape[2]] = decoder_attention.data[0, 0, :]

            topv, topi = decoder_output.data.topk(beam_size)

            for i in range(beam_size):
                p = topv[0][i]
                idp = topi[0][i]
                new_beam = [prob + p, route + [idp], decoder_hidden, atten]
                q.push(new_beam, new_beam[0])

        beams = [q.pop() for i in range(beam_size)]

        if beams[0][1][-1] == 1:
            break

    decoded_words = [lang.index2word[w] for w in beams[0][1][1:]]
    decoder_attentions = beams[0][3]
    return decoded_words, decoder_attentions[:len(decoded_words)]


def evaluate(encoder, decoder, valid_set, lang,
             embedding_size, encoder_style=ENCODER_STYLE, epoch_time=EPOCH_TIME,
             beam_size=1, verbose=True):
    valid_iter = data_iter(valid_set, batch_size=1, shuffle=True)
    if use_cuda:
        encoder.cuda()
        decoder.cuda()

    for iteration in range(epoch_time):
        data, idx_data = get_batch(next(valid_iter))
        rt, re, rm, summary = idx_data

        rt = Variable(torch.LongTensor(rt))
        re = Variable(torch.LongTensor(re))
        rm = Variable(torch.LongTensor(rm))

        summary = Variable(torch.LongTensor(summary))

        if use_cuda:
            rt, re, rm, summary = rt.cuda(), re.cuda(), rm.cuda(), summary.cuda()

        decoded_words, decoder_attentions = hierarchical_predictwords(rt, re, rm, summary,
                                                         encoder, decoder, lang,
                                                         embedding_size, encoder_style,
                                                         beam_size)

        res = ' '.join(decoded_words[:-1])
        if verbose:
            print(res)
        yield res


def setupconfig(args):
    parameters = {}
    for arg in vars(args):
        parameters[arg] = getattr(args, arg)
    print("---------------")
    print("Parameter Settings:")
    hierarchical_choices = ['HierarchicalRNN', 'HierarchicalBiLSTM',
                            'HierarchicalLIN']
    plain_choices = ['LIN', 'BiLSTM', 'RNN', 'BiLSTMMax']

    if parameters['encoder_style'] in hierarchical_choices and parameters['decoder_style'] != 'HierarchicalRNN':
        print("You must give me two hierarchical NNs!!!!!!!!!")
        quit()

    if parameters['encoder_style'] in plain_choices and parameters['decoder_style'] != 'RNN':
        print("You must give me two plain NNs!!!!!!!!!")
        quit()

    copy_player = COPY_PLAYER
    for arg in parameters:
        if arg == 'copy_player':
            if parameters[arg] == 'True':
                copy_player = True
        print("{} = {}".format(arg, parameters[arg]))
    print("---------------")
    parameters.pop('copy_player', None)

    return parameters, copy_player


def main(args):
    print("Start Training")

    parameters, copy_player = setupconfig(args)

    train_data, train_lang = loaddata(file_loc, 'train',
                                      copy_player=copy_player)
    if MAX_TRAIN_NUM is not None:
        train_data = train_data[:MAX_TRAIN_NUM]

    train_data = data2index(train_data, train_lang, max_sentences=parameters['max_sentence'])

    encoder, decoder = train(train_data, train_lang, **parameters)

    valid_data, _ = loaddata(file_loc, 'valid',
                             copy_player=copy_player)

    valid_data = data2index(valid_data, train_lang)
    evaluate(encoder, decoder, valid_data, train_lang['summary'],
             parameters['embedding_size'])


def parse_argument():
    encoder_choices = ['LIN', 'BiLSTM', 'RNN',
                       'BiLSTMMax', 'HierarchicalRNN',
                       'HierarchicalBiLSTM', 'HierarchicalLIN']

    decoder_choices = ['RNN', 'HierarchicalRNN']

    ap = argparse.ArgumentParser()
    ap.add_argument("-embed", "--embedding_size",
                    type=int, default=EMBEDDING_SIZE)

    ap.add_argument("-lr", "--learning_rate",
                    type=float, default=LR)

    ap.add_argument("-batch", "--batch_size",
                    type=int, default=BATCH_SIZE)

    ap.add_argument("-getloss", "--get_loss", type=int,
                    default=GET_LOSS)

    ap.add_argument("-encoder", "--encoder_style",
                    choices=encoder_choices, default=ENCODER_STYLE)

    ap.add_argument("-decoder", "--decoder_style",
                    choices=decoder_choices, default=DECODER_STYLE)

    ap.add_argument("-epochsave", "--save_model", type=int, default=SAVE_MODEL)

    ap.add_argument("-outputfile", "--output_file", default=OUTPUT_FILE)

    ap.add_argument("-copy", "--to_copy", choices=['True', 'False'],
                    default=TOCOPY)

    ap.add_argument("-copyplayer", "--copy_player", choices=['True', 'False'],
                    default=COPY_PLAYER)

    ap.add_argument("-gradclip", "--grad_clip", type=int, default=GRAD_CLIP)

    ap.add_argument("-pretrain", "--pretrain", default=PRETRAIN)

    ap.add_argument("-iternum", "--iter_num", default=iterNum)

    ap.add_argument("-layer", "--layer_depth", type=int, default=LAYER_DEPTH)

    ap.add_argument("-epoch", "--epoch_time", type=int, default=EPOCH_TIME)

    ap.add_argument("-maxlength", "--max_length", type=int, default=MAX_LENGTH)

    # max_sentence is optional
    ap.add_argument("-maxsentence", "--max_sentence", type=int, default=MAX_SENTENCES)

    return ap.parse_args()


if __name__ == '__main__':
    args = parse_argument()
    main(args)
