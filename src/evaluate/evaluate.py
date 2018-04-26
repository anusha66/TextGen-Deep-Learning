"""Evaluate the model."""
import sys
sys.path.append("../train")
from dataprepare import loaddata, data2index
from train import evaluate
from model import AttnDecoderRNN, EncoderBiLSTM, EncoderRNN, EncoderLIN, docEmbedding
from settings import file_loc
from util import load_model

import json
import os
import configparser
import argparse

config = configparser.ConfigParser()

train_data, train_lang = loaddata(file_loc, 'train')

embedding_size = 600
langs = train_lang
emb = docEmbedding(langs['rt'].n_words, langs['re'].n_words,
                   langs['rm'].n_words, embedding_size)
emb.init_weights()

encoder = EncoderLIN(embedding_size, emb)


def generate_text(model, data_file, output):
    encoder_src = model['encoder_path']
    decoder_src = model['decoder_path']
    encoder_style = None

    # Choose model architecture
    if 'RNN' in encoder_src:
        encoder = EncoderRNN(embedding_size, emb)
        encoder_style = 'RNN'
    elif 'LSTM' in encoder_src:
        encoder = EncoderBiLSTM(embedding_size, emb)
        encoder_style = 'LSTM'
    else:
        encoder = EncoderLIN(embedding_size, emb)
        encoder_style = 'LIN'

    decoder = AttnDecoderRNN(embedding_size, langs['summary'].n_words)
    encoder = load_model(encoder, encoder_src)
    decoder = load_model(decoder, decoder_src)
    data_path = os.path.join(data_file['data_dir'], data_file['data_name'] + '.json')
    with open(data_path) as f:
        valuation_data = json.load(f)
    assert valuation_data is not None

    valid_data, _ = loaddata(data_file['data_dir'], data_file['data_name'])
    data_length = len(valid_data)
    valid_data = data2index(valid_data, train_lang)
    text_generator = evaluate(encoder, decoder, valid_data,
                              train_lang['summary'], embedding_size,
                              encoder_style=encoder_style, iter_time=data_length,
                              beam_size=1, verbose=False)
    print('The text generation begin\n', flush=True)
    with open(output, 'w') as f:
        for idx, line in enumerate(text_generator):
            print('Summery generated, No{}'.format(idx + 1))
            f.write(line + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main for generating the text for extract.')
    parser.add_argument('-config', type=str, default="config.cfg",
                        help="path to config file.")
    args = parser.parse_args()
    config.read(args.config)
    for section in config.sections():
        if 'evaluate' not in section:
            continue
        model, data_file = {}, {}
        model['encoder_path']  = config.get(section, 'encoder_path')
        model['decoder_path']  = config.get(section, 'decoder_path')
        data_file['data_dir']  = config.get(section, 'data_dir')
        data_file['data_name'] = config.get(section, 'data_name')
        output                 = config.get(section, 'output')
        print("{} has started\n".format(section), flush=True)
        generate_text(model, data_file, output)
        print("{} has been done\n".format(section), flush=True)
