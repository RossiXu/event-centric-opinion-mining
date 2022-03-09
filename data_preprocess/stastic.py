import argparse

import jieba

events = []
titles = []
contents = []


def get_spans(text, language='english'):
    if language == 'chinese':
        words = list(jieba.cut(text))
    else:
        words = text.split()
    return words


def data_preprocess(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = f.read()
    psgs = data.split('-' * 54)
    insts = []
    for psg in psgs:
        psg = psg.strip().split('\n')
        event = psg[0].split('\t')[0]
        lines = psg[2:]
        opinion = []
        for line_idx, line in enumerate(lines):
            if len(line.strip().split('\t')) == 3:
                sent, label, aspect = [ele.strip() for ele in line.strip().split('\t') if len(ele)]
            else:
                continue
            if (label == 'B' and not opinion) or label == 'I':
                opinion.append(sent)
            elif label == 'B' and opinion:
                old_aspect = [ele for ele in lines[line_idx - 1].strip().split('\t') if len(ele)][2]
                insts.append([opinion, event, old_aspect])
                opinion = [sent]
            elif label == 'O' and opinion:
                old_aspect = [ele for ele in lines[line_idx-1].strip().split('\t') if len(ele)][2]
                insts.append([opinion, event, old_aspect])
                opinion = []
        if len(opinion):
            insts.append([opinion, event, aspect])
    # print(file_name, 'Num of opinions:', len(insts), sep='\t')
    return insts


def parse_arguments(parser):
    # Training hyper parameters
    parser.add_argument('--data_dir', type=str, default='data/ECOB-ZH/')
    parser.add_argument('--data_file', type=str, default="data.txt")
    parser.add_argument('--language', type=str, default='chinese')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    Data Statistics.
    """
    # Parse arguments.
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    file_name = args.data_dir + args.data_file
    lang = args.language

    # get docs: events, titles, contents
    with open(file_name, 'r', encoding='utf-8') as f:
        data = f.read()
        documents = data.split('-' * 54)
        documents = [doc for doc in documents if len(doc.strip())]
        for document in documents:
            document = document.strip()
            if document:
                lines = document.split('\n')
                events.append(lines[0].strip().split('\t')[0])
                titles.append(lines[1].strip())
                contents.append([line.strip().split('\t') for line in lines[2:]])
    # get opinions: opinion_text, event, aspect
    opinions = data_preprocess(file_name)

    event_words = [get_spans(event, language=lang) for event in events]
    event_lens = [len(event_word) for event_word in event_words]

    # doc data
    doc_lens = [len(content) for content in contents]
    print('Document Number: ', len(documents))
    print('Document Avg. Sents: ', sum(doc_lens) / len(events))
    print('Document Avg. opinion: ', len(opinions) / len(doc_lens))

    # opinion data
    opinion_lens = [len(opinion[0]) for opinion in opinions]
    print('Opinion Number: ', len(opinion_lens))
    print('opinion ratio: ', sum(opinion_lens) / sum(doc_lens))
    print('Opinion avg len: ', sum(opinion_lens) / len(opinion_lens), sum(opinion_lens))

    # basic data
    print('Event Number: ', len(list(set(events))))
    print('Event Avg. Tokens: ', sum(event_lens) / len(events))