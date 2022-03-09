import argparse


def parse_arguments(parser):
    # Training hyper parameters
    parser.add_argument('--result_dir', type=str, default='result/chinese_result/')  # Root data directory.
    parser.add_argument('--result_file', type=str, default='seq.results')
    parser.add_argument('--data_dir', type=str, default='data/ECOB-ZH/')
    parser.add_argument('--gold_file', type=str, default='test.txt')
    parser.add_argument('--language', type=str, default='')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    result_dir = args.result_dir
    result_file = args.result_file
    data_dir = args.data_dir
    gold_file = args.gold_file
    language = args.language

    opinion_level = 'segment'
    max_opinion_len = 5

    # Read result file.
    with open(result_dir + result_file, 'r', encoding='utf-8') as f:
        data = f.read()
        psgs = data.split('-' * 54)
        new_psgs = []
        for psg in psgs:
            lines = psg.strip().split('\n')
            if len(lines) <= 2:
                continue
            content = lines[2:]
            new_psg = []
            new_psg.append(lines[0])
            new_psg.append(lines[1])
            for idx, line in enumerate(content):
                sent, pred_label, aspect, gold_label = [ele.strip() for ele in line.strip().split('\t') if len(ele)]
                new_psg.append([sent, gold_label, pred_label, aspect])
            new_psgs.append(new_psg)
    gold_opinions = []
    pred_opinions = []
    for new_psg in new_psgs:
        gold_span = []
        pred_span = []
        gold_start, gold_end = -1, -1
        pred_start, pred_end = -1, -1
        lines = new_psg[2:]
        # event and title
        pred_psg = []
        pred_psg.append(new_psg[0])
        pred_psg.append(new_psg[1])
        # content
        for line in lines:
            pred_psg.append([line[0], line[2], 'O', line[1], line[3]])
        for idx, line in enumerate(lines):
            # get gold span
            if line[1] == 'B':
                if gold_start >= 0 and gold_end >= 0 and lines[idx-1][1] != 'O':
                    gold_span.append(lines[gold_start:gold_end+1])
                gold_start = idx
                gold_end = idx
            elif line[1] == 'I':
                gold_end = idx
            elif line[1] == 'O':
                if gold_start >= 0 and gold_end >= 0 and lines[idx-1][1] != 'O':
                    gold_span.append(lines[gold_start:gold_end+1])

            # get pred span
            if line[2] == 'B':
                if pred_end >= 0 and pred_end >= 0 and lines[idx-1][2] != 'O':
                    pred_span.append(lines[pred_start:pred_end+1])
                pred_start = idx
                pred_end = idx
            elif line[2] == 'I':
                pred_end = idx
            elif line[2] == 'O':
                if pred_start >= 0 and pred_end >= 0 and lines[idx-1][2] != 'O':
                    pred_span.append(lines[pred_start:pred_end+1])
        # rest span
        if gold_start >= 0 and gold_end >= 0 and lines[-1][1] != 'O':
            gold_span.append(lines[gold_start:gold_end+1])
        if pred_start >= 0 and pred_end >= 0 and lines[-1][2] != 'O':
            pred_span.append(lines[pred_start:pred_end+1])
        gold_span = [[sent[0] for sent in s] for s in gold_span]
        pred_span = [[sent[0] for sent in s] for s in pred_span]
        pred_sentences = []
        for sents in pred_span:
            pred_sentences.extend(sents)

        # calculate
        for idx, gold_s in enumerate(gold_span):
            opinion_len = len(gold_s)
            if 'e' in language:
                opinion_word_num = sum([len(s.strip().split()) for s in gold_s])
            else:
                opinion_word_num = sum([len(s) for s in gold_s])
            corr_num = 0
            for s in gold_s:
                if s in pred_sentences:
                    corr_num += 1
            corr = 0
            for pred_s in pred_span:
                if set(gold_s) == set(pred_s):
                    corr = 1
            gold_opinions.append([opinion_len, opinion_word_num, corr_num, ''.join(gold_s), corr])

    wrong_opinions = []

    print('Opinion_Len', 'Correct_Rate', 'Partially_Correct_Rate', 'Wrong_Rate')
    for opinion_len in range(1, max_opinion_len+1):
        choose_opinions = []
        for gold_opinion in gold_opinions:
            if gold_opinion[0] == opinion_len:
                choose_opinions.append(gold_opinion)
            if opinion_len == max_opinion_len and gold_opinion[0] > opinion_len:
                choose_opinions.append(gold_opinion)
        total_corr = 0
        part_corr = 0
        total_wrong = 0
        for opinion in choose_opinions:
            if opinion[0] == opinion[2] and opinion[4] == 1:
                total_corr += 1
            elif opinion[2] == 0:
                total_wrong += 1
                wrong_opinion = [str(opinion_len)]
                for ele in opinion:
                    wrong_opinion.append(str(ele))
                wrong_opinions.append(wrong_opinion)
            elif opinion[2] != 0 and (opinion[2] < opinion[0] or opinion[4] == 0):
                part_corr += 1
        if opinion_len == 1:
            total_wrong += part_corr
            part_corr = 0
        total_opinion = part_corr + total_corr + total_wrong
        print(opinion_len, int(total_corr/total_opinion*100), int(part_corr/total_opinion*100), int(total_wrong/total_opinion*100))

    # with open('wrong_opinions.txt', 'w', encoding='utf-8') as f:
    #     for opinion in wrong_opinions:
    #         f.write('\t'.join(opinion) + '\n')