import argparse

import jieba


def get_clear_text(text):
    text = text.strip().split()
    return ''.join(text)


def parse_arguments(parser):
    # Training hyper parameters
    parser.add_argument('--data_dir', type=str, default='data/ECOB-ZH/')  # Root data directory.
    parser.add_argument('--language', type=str, default='chinese')  # language: ['english', 'chinese']
    parser.add_argument('--result_dir', type=str, default='result/chinese_result/')
    parser.add_argument('--result_file', type=str, default='mrc.results')
    parser.add_argument('--downstream_model', type=str, default='mrc')  # ['enum', 'mrc'].
    parser.add_argument('--aspect_level', type=str, default='total')  # ['subevent', 'event', 'entity', 'total']

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    Evaluate performance of Enum and MRC model on OTE.
    """
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    data_dir = args.data_dir
    language = args.language
    result_dir = args.result_dir
    result_file = result_dir + args.result_file
    model = args.downstream_model
    aspect_level = args.aspect_level

    if model == 'enum':
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                aspect_opinions = [[line.strip().split('\t')[0], line.strip().split('\t')[2], line.strip().split('\t')[1]] for line in lines if len(line.strip().split('\t')) == 3]  # opinion text, pred aspect
        except FileNotFoundError:
            print('There is no ' + result_file)

    if model == 'mrc':
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # opinion, pred_aspect, gold_aspect, event
                aspect_opinions = [[line.strip().split('\t')[3], line.strip().split('\t')[2], line.strip().split('\t')[1], line.strip().split('\t')[0]] for line in lines]
        except FileNotFoundError:
            print('There is no ' + result_file)

    # get_aspect
    if aspect_level == 'subevent':
        with open(data_dir + 'subevent_aspect.txt', 'r', encoding='utf-8') as f:
            subevent_aspects = [line.strip() for line in f.readlines()]
        aspects = subevent_aspects
    elif aspect_level == 'entity':
        with open(data_dir + 'entity_aspect.txt', 'r', encoding='utf-8') as f:
            entity_aspects = [line.strip() for line in f.readlines()]
        aspects = entity_aspects
    elif aspect_level == 'event':
        with open(data_dir + 'event_aspect.txt', 'r', encoding='utf-8') as f:
            event_aspects = [line.strip() for line in f.readlines()]
        aspects = event_aspects
    elif aspect_level == 'total':
        aspects = []
        with open(data_dir + 'data.txt', 'r', encoding='utf-8') as f:
            data = f.read()
            psgs = data.split('-' * 54)
            for psg in psgs:
                lines = psg.strip().split('\n')
                for line in lines[2:]:
                    sent, label, aspect = line.strip().split('\t')
                    aspects.append(aspect)
    aspects = list(set([get_clear_text(a) for a in aspects]))

    total_opinion = 0
    corr_opinion = 0
    for opinion in aspect_opinions:
        if get_clear_text(opinion[2].strip()) in aspects:
            if opinion[2].strip() == 'O':
                continue
            if get_clear_text(opinion[1].strip()) == get_clear_text(opinion[2].strip()):
                corr_opinion += 1
            total_opinion += 1
    print('Accuracy: ', '%.4f' % (corr_opinion / total_opinion))

    ground_word_num = 0
    pred_word_num = 0
    overlap_word_num = 0
    for opinion in aspect_opinions:
        if language == 'english':
            pred_aspect = opinion[1].strip().split()
            gold_aspect = opinion[2].strip().split()
        else:
            pred_aspect = list(jieba.cut(opinion[1]))
            gold_aspect = list(jieba.cut(opinion[2]))
        overlap_word = [i for i in pred_aspect if i in gold_aspect]
        overlap_word_num += len(overlap_word)
        ground_word_num += len(gold_aspect)
        pred_word_num += len(pred_aspect)
    r = overlap_word_num / ground_word_num
    p = overlap_word_num / pred_word_num
    f = 2 * r * p / (r + p)
    print('Overlap-F1: ', '%.4f' % f)


