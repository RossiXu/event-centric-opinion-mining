import argparse


def get_clear_text(text):
    text = text.strip().split()
    return ''.join(text)


def get_segment_result(downstream_model='gold'):
    aspect_opinions = []
    model = 'gold'
    if downstream_model == 'enum':
        try:
            with open(result_dir + downstream_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                aspect_opinions = [[line.strip().split('\t')[0], line.strip().split('\t')[2]] for line in lines]  # opinion text, gold aspect
                model = 'enum'
        except FileNotFoundError:
            print('There is no ' + result_dir + downstream_file)

    if downstream_model == 'mrc':
        try:
            with open(result_dir + downstream_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                aspect_opinions = [[line.strip().split('\t')[3], line.strip().split('\t')[2]] for line in lines]
                model = 'mrc'
        except FileNotFoundError:
            print('There is no ' + result_dir + downstream_file)

    with open(result_file, 'r', encoding='utf-8') as f:
        pair_opinions = []
        lines = [line.strip().split('\t') for line in f.readlines()]
        for line in lines:
            if int(line[3]) == 1:
                pair_opinions.append(line[1])

    with open(data_dir + test_file, 'r', encoding='utf-8') as f:
        data = f.read()
        psgs = data.strip().split('-'*54)
        new_psgs = []
        for psg in psgs:
            lines = psg.strip().split('\n')
            if len(lines) <= 2:
                continue
            event = lines[0].strip()
            title = lines[1].strip()
            new_psg = [event, title]
            for idx, line in enumerate(lines[2:]):
                sent, gold_label, aspect = [ele.strip() for ele in line.strip().split('\t') if len(ele)]
                if sent not in pair_opinions:
                    pred_label = 'O'
                elif sent in pair_opinions and idx == 0:
                    pred_label = 'B'
                elif sent in pair_opinions and idx > 0:
                    if pred_label == 'O':
                        pred_label = 'B'
                    else:
                        pred_label = 'I'
                new_psg.append([sent, gold_label, pred_label, aspect])
            new_psgs.append(new_psg)

    gold_num = 0
    pred_num = 0
    corr_num = 0
    pred_psgs = []
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
                    gold_span.append([gold_start, gold_end, lines[idx-1][3]])
                gold_start = idx
                gold_end = idx
            elif line[1] == 'I':
                gold_end = idx
            elif line[1] == 'O':
                if gold_start >= 0 and gold_end >= 0 and lines[idx-1][1] != 'O':
                    gold_span.append([gold_start, gold_end, lines[idx-1][3]])

            # get pred span
            if line[2] == 'B':
                if pred_end >= 0 and pred_end >= 0 and lines[idx-1][2] != 'O':
                    pred_span.append([pred_start, pred_end])
                pred_start = idx
                pred_end = idx
            elif line[2] == 'I':
                pred_end = idx
            elif line[2] == 'O':
                if pred_start >= 0 and pred_end >= 0 and lines[idx-1][2] != 'O':
                    pred_span.append([pred_start, pred_end])
        # rest span
        if gold_start >= 0 and gold_end >= 0 and lines[-1][1] != 'O':
            gold_span.append([gold_start, gold_end, lines[-1][3]])
        if pred_start >= 0 and pred_end >= 0 and lines[-1][2] != 'O':
            pred_span.append([pred_start, pred_end])

        # get pred aspect
        for idx, pred_single_span in enumerate(pred_span):
            text = [ele[0] for ele in lines[pred_single_span[0]:pred_single_span[1]+1]]
            text = ''.join(text)
            for opinion in aspect_opinions:
                if opinion[0] == text:
                    # update pred psg
                    for i in range(pred_single_span[0], pred_single_span[1]+1):
                        pred_psg[i+2][2] = opinion[1]
                    pred_span[idx].append(opinion[1])
                    break

        # calculate
        gold_span_num = len(gold_span)
        pred_span_num = len(pred_span)
        correct_span = []
        for gold_single_span in gold_span:
            for pred_single_span in pred_span:
                if downstream_model == 'gold':
                    if gold_single_span[0] == pred_single_span[0] and gold_single_span[1] == pred_single_span[1]:
                        correct_span.append(gold_single_span)
                        break
                else:
                    if len(gold_single_span) == len(pred_single_span) and gold_single_span[0] == pred_single_span[0] and \
                            gold_single_span[1] == pred_single_span[1] and get_clear_text(gold_single_span[2]) == get_clear_text(pred_single_span[2]):
                        correct_span.append(gold_single_span)
                        break
        correct_span_num = len(correct_span)

        gold_num += gold_span_num
        pred_num += pred_span_num
        corr_num += correct_span_num

        pred_psgs.append(pred_psg)

    try:
        p = corr_num / pred_num
        r = corr_num / gold_num
        f = 2 * p * r / (p + r)
        print('F1 score: ', '%.4f' % p, '%.4f' % r, '%.4f' % f)
    except ZeroDivisionError:
        print('F1 score: ', 0)

    with open(result_dir + 'paircls_' + model + '.result', 'w', encoding='utf-8') as f:
        for pred_psg in pred_psgs:
            f.write(pred_psg[0] + '\n')
            f.write(pred_psg[1] + '\n')
            for line in pred_psg[2:]:
                f.write(line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + line[3] + '\t' + line[4] + '\t' + '\n')
            f.write('-'*54 + '\n')


def get_sent_result(downstream_model='gold'):
    aspect_opinions = []
    model = 'gold'
    if downstream_model == 'enum':
        try:
            with open(result_dir + downstream_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                aspect_opinions = [[line.strip().split('\t')[0], line.strip().split('\t')[2]] for line in
                                   lines]  # opinion text, gold aspect
                model = 'enum'
        except FileNotFoundError:
            print('There is no ' + result_dir + downstream_file)

    if downstream_model == 'mrc':
        try:
            with open(result_dir + downstream_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                aspect_opinions = [[line.strip().split('\t')[3], line.strip().split('\t')[2]] for line in lines]
                model = 'mrc'
        except FileNotFoundError:
            print('There is no ' + result_dir + downstream_file)

    # Read result_file.
    with open(result_file, 'r', encoding='utf-8') as f:
        pair_opinions = []  # Predicted opinion sentences.
        lines = [line.strip().split('\t') for line in f.readlines()]
        for line in lines:
            if int(line[3]) == 1:
                pair_opinions.append(line[1])

    # Read test.txt.
    with open(data_dir + test_file, 'r', encoding='utf-8') as f:
        data = f.read()
        psgs = data.strip().split('-'*54)
        new_psgs = []
        for psg in psgs:
            lines = psg.strip().split('\n')
            if len(lines) <= 2:
                continue
            event = lines[0].strip()
            title = lines[1].strip()
            new_psg = [event, title]
            for idx, line in enumerate(lines[2:]):
                sent, gold_label, aspect = [ele.strip() for ele in line.strip().split('\t') if len(ele)]
                if sent in pair_opinions:
                    pred_label = 'B'
                else:
                    pred_label = 'O'
                if gold_label == 'I':
                    gold_label = 'B'
                new_psg.append([sent, gold_label, pred_label, aspect])
            new_psgs.append(new_psg)

    gold_num = 0
    pred_num = 0
    corr_num = 0
    pred_psgs = []
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
                    gold_span.append([gold_start, gold_end, lines[idx-1][3]])
                gold_start = idx
                gold_end = idx
            elif line[1] == 'I':
                gold_end = idx
            elif line[1] == 'O':
                if gold_start >= 0 and gold_end >= 0 and lines[idx-1][1] != 'O':
                    gold_span.append([gold_start, gold_end, lines[idx-1][3]])

            # get pred span
            if line[2] == 'B':
                if pred_end >= 0 and pred_end >= 0 and lines[idx-1][2] != 'O':
                    pred_span.append([pred_start, pred_end])
                pred_start = idx
                pred_end = idx
            elif line[2] == 'I':
                pred_end = idx
            elif line[2] == 'O':
                if pred_start >= 0 and pred_end >= 0 and lines[idx-1][2] != 'O':
                    pred_span.append([pred_start, pred_end])
        # rest span
        if gold_start >= 0 and gold_end >= 0 and lines[-1][1] != 'O':
            gold_span.append([gold_start, gold_end, lines[-1][3]])
        if pred_start >= 0 and pred_end >= 0 and lines[-1][2] != 'O':
            pred_span.append([pred_start, pred_end])

        # get pred aspect
        for idx, pred_single_span in enumerate(pred_span):
            text = [ele[0] for ele in lines[pred_single_span[0]:pred_single_span[1]+1]]
            text = ''.join(text)
            for opinion in aspect_opinions:
                if get_clear_text(text) in get_clear_text(opinion[0]):
                    # update pred psg
                    for i in range(pred_single_span[0], pred_single_span[1]+1):
                        pred_psg[i+2][2] = opinion[1]
                    pred_span[idx].append(opinion[1])
                    break

        # Calculate.
        gold_span_num = len(gold_span)
        pred_span_num = len(pred_span)
        correct_span = []
        for gold_single_span in gold_span:
            for pred_single_span in pred_span:
                if downstream_model == 'gold':
                    if gold_single_span[0] == pred_single_span[0] and gold_single_span[1] == pred_single_span[1]:
                        correct_span.append(gold_single_span)
                        break
                else:
                    if len(gold_single_span) == len(pred_single_span) and gold_single_span[0] == pred_single_span[0] and \
                            gold_single_span[1] == pred_single_span[1] and get_clear_text(gold_single_span[2]) == get_clear_text(pred_single_span[2]):
                        correct_span.append(gold_single_span)
                        break
        correct_span_num = len(correct_span)

        gold_num += gold_span_num
        pred_num += pred_span_num
        corr_num += correct_span_num

        pred_psgs.append(pred_psg)

    try:
        p = corr_num / pred_num
        r = corr_num / gold_num
        f = 2 * p * r / (p + r)
        print('F1 score: ', '%.4f' % p, '%.4f' % r, '%.4f' % f)
    except ZeroDivisionError:
        print('F1 score: ', 0)

    with open(result_dir + 'pair_' + model + '_sent.result', 'w', encoding='utf-8') as f:
        for pred_psg in pred_psgs:
            f.write(pred_psg[0] + '\n')
            f.write(pred_psg[1] + '\n')
            for line in pred_psg[2:]:
                f.write(line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + line[3] + '\t' + line[4] + '\t' + '\n')
            f.write('-'*54 + '\n')


def parse_arguments(parser):
    # Training hyper parameters
    parser.add_argument('--data_dir', type=str, default='data/ECOB-ZH/')  # Root data directory.
    parser.add_argument('--test_file', type=str, default='test.txt')
    parser.add_argument('--result_dir', type=str, default='result/chinese_result/')
    parser.add_argument('--result_file', type=str, default='pair_classification.results')
    parser.add_argument('--downstream_model', type=str, default='mrc')  # Evaluate Seq model on EOT if choose 'gold', evaluate Seq-Enum/MRC pipeline if 'enum'/'mrc'.
    parser.add_argument('--downstream_file', type=str, default='enumerate.results')  # Result file of downstream model: ['enumerate.results', 'bert_sqad.txt']
    parser.add_argument('--opinion_level', type=str, default='all')  # ['all', 'segment', 'sent']

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    Evaluate performance of PairCls model.
    """
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    data_dir = args.data_dir
    test_file = args.test_file
    result_dir = args.result_dir
    result_file = args.result_dir + args.result_file
    downstream_model = args.downstream_model
    downstream_file = args.downstream_file

    if args.opinion_level == 'all' or args.opinion_level == 'segment':
        print('Segment-F1: ')
        get_segment_result(downstream_model)
    if args.opinion_level == 'all' or args.opinion_level == 'sent':
        print('Sent-F1: ')
        get_sent_result(downstream_model)