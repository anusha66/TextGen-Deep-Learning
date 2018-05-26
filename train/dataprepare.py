from preprocessing import readfile
from settings import file_loc, MAX_SENTENCES, PLAYER_PADDINGS, COPY_PLAYER


class Lang:

    def __init__(self, name):
        self.name = name
        self.word2index = {"<SOS>": 0, "<EOS>": 1, "<PAD>": 2, "<UNK>": 3, "<EOB>": 4, "<BLK>": 5}
        self.word2count = {"<SOS>": 0, "<EOS>": 0, "<PAD>": 0, "<UNK>": 0, "<EOB>": 0, "<BLK>": 0}
        self.index2word = {0: "<SOS>", 1: "<EOS>", 2: "<PAD>", 3: "<UNK>", 4: "<EOB>", 5: "<BLK>"}
        self.n_words = len(self.word2index)

    def addword(self, word):
        if word in PLAYER_PADDINGS:
            word = "<PAD>"

        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def readLang(data_set):
    rt = Lang('rt')
    re = Lang('re')
    rm = Lang('rm')
    summarize = Lang('summarize')

    for v in data_set:
        for triplet in v.triplets:
            rt.addword(triplet[0])
            re.addword(triplet[1])
            rm.addword(triplet[2])
            summarize.addword(triplet[2])
    for v in data_set:
        for word in v.summary:
            summarize.addword(word)

    return rt, re, rm, summarize


def loaddata(data_dir, mode='train', max_len=None, copy_player=COPY_PLAYER):
    data_set = readfile(data_dir + mode + '.json', copy_player=copy_player)
    if max_len is not None:
        data_set = data_set[:max_len]
    rt, re, rm, summary = readLang(data_set)

    print("Read %s data" % mode)
    print("Read %s box score summary" % len(data_set))
    print("Embedding size of (r.t, r.e, r.m) and summary:")
    print("({}, {}, {}), {}".format(rt.n_words, re.n_words,
                                    rm.n_words, summary.n_words))

    langs = {'rt': rt, 're': re, 'rm': rm, 'summary': summary}
    return data_set, langs


def data2index(data_set, langs, max_sentences=MAX_SENTENCES):
    def findword2index(lang, word):
        try:
            return lang.word2index[word]
        except KeyError:
            return lang.word2index['<UNK>']

    for i in range(len(data_set)):
        idx_triplets = []
        for triplet in data_set[i].triplets:
            idx_triplet = [None, None, None]
            idx_triplet[0] = findword2index(langs['rt'], triplet[0])
            idx_triplet[1] = findword2index(langs['re'], triplet[1])
            idx_triplet[2] = findword2index(langs['rm'], triplet[2])
            idx_triplets.append(tuple(idx_triplet))

        idx_summary = []
        sentence_cnt = 0
        for word in data_set[i].summary:
            idx_summary.append(findword2index(langs['summary'], word))

            if word == '.':
                sentence_cnt += 1

            if max_sentences is not None and sentence_cnt >= max_sentences:
                break

        data_set[i].idx_data = [idx_triplets] + [idx_summary]
        data_set[i].sent_leng = sentence_cnt

    return data_set


def showsentences(dataset):
    for t in dataset:
        for w in t[1]:
            if w == '.':
                print('')
            else:
                print(w, end=' ')

