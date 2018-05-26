import random
import json
from pprint import pprint

from settings import file_loc, MAX_PLAYERS, COPY_PLAYER

class Boxdata:
    def __init__(self):
        self.triplets = []
        self.summary = []
        self.idx_data = []
        self.sent_leng = 0


def readfile(filename, copy_player=COPY_PLAYER):
    def add_blocks(tokens):
        summary = []
        for token in tokens:
            summary.append(token)
            if token is '.':
                summary.append('<BLK>')
        summary.append('<EOS>')
        return summary

    result = []
    with open(filename, 'r') as f:
        data = json.load(f)
        for d in data:
            d = align_box_score(d)

            boxdata = Boxdata()
            boxdata.triplets = doc2vec(d, copy_player=copy_player)
            boxdata.summary = add_blocks(d['summary'])
            result.append(boxdata)
    return result


def doc2vec(doc, copy_player=COPY_PLAYER):
    triplets = []

    def maketriplets(doc, key, ignore, title):
        new_triplets = []
        for _type, _type_dic in doc[key].items():

            if _type in ignore:
                continue
            if key == 'box_score':
                for num, value in _type_dic.items():
                    entity = doc[key][title][num]
                    if entity in ['<PAD>', '<EOB>']:
                        new_triplets.append((entity, entity, entity))
                    else:
                        new_triplets.append((_type, entity, value))
            else:
                entity = doc[key][title]
                new_triplets.append((_type, entity, _type_dic))
        return new_triplets

    keys = ['box_score', 'home_name', 'home_city', 'vis_name',
            'vis_city', 'home_line', 'vis_line']

    for k in keys:
        if k == 'box_score':
            if copy_player:
                ignore = []
            else:
                ignore = ['FIRST_NAME', 'SECOND_NAME', 'PLAYER_NAME']
            title = 'PLAYER_NAME'
            new_triplets = maketriplets(doc, k, ignore, title)
            triplets += new_triplets

        elif k in ['vis_line', 'home_line']:
            ignore = ['TEAM-NAME']
            title = 'TEAM-NAME'
            new_triplets = maketriplets(doc, k, ignore, title)
            triplets += new_triplets

        else:
            if 'name' in k:
                new_triplets = [('name', k, doc[k])]
            elif 'city' in k:
                new_triplets = [('city', k, doc[k])]
            else:
                continue
            triplets += new_triplets

    return triplets


def test_box_score_aligned(d):
    tables = 0
    EB_ATTR = 0
    for k in d['box_score']:
        tables = tables + 1
        if 'ENDBLOCK' in d['box_score']:
            EB_ATTR = EB_ATTR + 1
        if len(d['box_score'][k]) != MAX_PLAYERS or tables != EB_ATTR:
            print("Preprocessing Error")
            pprint(d['box_score'][k])
    return


def data_iter(source, batch_size=32, shuffle=True):
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    if shuffle:
        random.shuffle(source)

    source.sort(key=lambda boxdata: boxdata.sent_leng)
    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            return

        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        yield batch


def align_box_score(doc):
    NULL_PAD = '<PAD>'
    END_OF_BLOCK = '<EOB>'
    NULL_ENTITIES = []
    ignore = ['PLAYER_NAME', 'FIRST_NAME', 'SECOND_NAME']
    TEAM_SIZE = len(doc['box_score']['PLAYER_NAME'])
    if TEAM_SIZE < MAX_PLAYERS:
        NULL_ENTITIES = ['<PAD' + str(i) + '>' for i in range(MAX_PLAYERS - TEAM_SIZE)]
        for i in range(MAX_PLAYERS - TEAM_SIZE):
            doc['box_score']['PLAYER_NAME'][NULL_ENTITIES[i]] = NULL_PAD
            doc['box_score']['FIRST_NAME'][NULL_ENTITIES[i]] = NULL_PAD
            doc['box_score']['SECOND_NAME'][NULL_ENTITIES[i]] = NULL_PAD

    doc['box_score']['PLAYER_NAME']['ENDBLOCK'] = END_OF_BLOCK
    doc['box_score']['FIRST_NAME']['ENDBLOCK'] = END_OF_BLOCK
    doc['box_score']['SECOND_NAME']['ENDBLOCK'] = END_OF_BLOCK
    for attr, val in doc['box_score'].items():
        if attr in ignore:
            continue
        if len(val) < MAX_PLAYERS:
            for i in range(MAX_PLAYERS - len(val)):
                val[NULL_ENTITIES[i]] = NULL_PAD
        val['ENDBLOCK'] = END_OF_BLOCK

    return doc

