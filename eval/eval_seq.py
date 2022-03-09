import argparse


def get_clear_text(text):
    text = text.strip().split()
    return ''.join(text)


def get_sent_result():
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
        data = f.read()
        psgs = data.split('-'*54)
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

    # get opinion extraction precision
    gold_opinion = 0
    pred_opinion = 0
    corr_opinion = 0
    total_sent = 0
    corr_sent = 0
    for new_psg in new_psgs:
        for line in new_psg[2:]:
            if line[1] != 'O':
                gold_opinion += 1
            if line[2] != 'O':
                pred_opinion += 1
            if line[1] != 'O' and line[2] != 'O':
                corr_opinion += 1
            total_sent += 1
            if (line[1] != 'O' and line[2] != 'O') or (line[1] == 'O' and line[2] == 'O'):
                corr_sent += 1
    # p = corr_opinion / pred_opinion
    # r = corr_opinion / gold_opinion
    # f = 2 * p * r / (p + r)
    # print('Sent level Opinion Extraction: ', p, r, f, corr_opinion, pred_opinion, gold_opinion, sep='\t')
    # print('Sent level Opinion Extraction Precision', corr_sent / total_sent, corr_sent, total_sent, sep='\t')

    # get final result
    for psg_idx, new_psg in enumerate(new_psgs):
        for line_idx, line in enumerate(new_psg[2:]):
            if line[1] == 'I':
                new_psgs[psg_idx][line_idx+2][1] = 'B'
            if line[2] == 'I':
                new_psgs[psg_idx][line_idx+2][2] = 'B'

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
                if gold_start >= 0 and gold_end >= 0 and lines[idx - 1][1] != 'O':
                    gold_span.append([gold_start, gold_end, lines[idx - 1][3]])
                gold_start = idx
                gold_end = idx
            elif line[1] == 'I':
                gold_end = idx
            elif line[1] == 'O':
                if gold_start >= 0 and gold_end >= 0 and lines[idx - 1][1] != 'O':
                    gold_span.append([gold_start, gold_end, lines[idx - 1][3]])

            # get pred span
            if line[2] == 'B':
                if pred_end >= 0 and pred_end >= 0 and lines[idx - 1][2] != 'O':
                    pred_span.append([pred_start, pred_end])
                pred_start = idx
                pred_end = idx
            elif line[2] == 'I':
                pred_end = idx
            elif line[2] == 'O':
                if pred_start >= 0 and pred_end >= 0 and lines[idx - 1][2] != 'O':
                    pred_span.append([pred_start, pred_end])
        # rest span
        if gold_start >= 0 and gold_end >= 0 and lines[-1][1] != 'O':
            gold_span.append([gold_start, gold_end, lines[-1][3]])
        if pred_start >= 0 and pred_end >= 0 and lines[-1][2] != 'O':
            pred_span.append([pred_start, pred_end])

        # get pred aspect
        for idx, pred_single_span in enumerate(pred_span):
            text = [ele[0] for ele in lines[pred_single_span[0]:pred_single_span[1] + 1]]
            text = ''.join(text)
            for opinion in aspect_opinions:
                if get_clear_text(text.strip()) in get_clear_text(opinion[0].strip()):
                    # update pred psg
                    for i in range(pred_single_span[0], pred_single_span[1] + 1):
                        pred_psg[i + 2][2] = opinion[1]
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

    with open(result_dir + 'seq_' + model + '_sent.result', 'w', encoding='utf-8') as f:
        for pred_psg in pred_psgs:
            f.write(pred_psg[0] + '\n')
            f.write(pred_psg[1] + '\n')
            for line in pred_psg[2:]:
                f.write(line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + line[3] + '\t' + line[4] + '\t' + '\n')
            f.write('-' * 54 + '\n')


def get_segment_result():
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

    with open(result_file, 'r', encoding='utf-8') as f:
        data = f.read()
        psgs = data.split('-'*54)
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
                if get_clear_text(opinion[0]) == get_clear_text(text):
                    # update pred psg
                    for i in range(pred_single_span[0], pred_single_span[1]+1):
                        pred_psg[i+2][2] = opinion[1]
                    if get_clear_text(opinion[1]) in aspects:
                        pred_span[idx].append(opinion[1])
                    else:
                        pred_span[idx] = None

        gold_span = [ele for ele in gold_span if get_clear_text(ele[2]) in aspects]
        pred_span = [ele for ele in pred_span if ele]

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

    # Save result file
    with open(result_dir + 'seq_' + model + '.result', 'w', encoding='utf-8') as f:
        for pred_psg in pred_psgs:
            f.write(pred_psg[0] + '\n')
            f.write(pred_psg[1] + '\n')
            for line in pred_psg[2:]:
                f.write(line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + line[3] + '\t' + line[4] + '\t' + '\n')
            f.write('-'*54 + '\n')


def parse_arguments(parser):
    # Training hyper parameters
    parser.add_argument('--data_dir', type=str, default='data/ECOB-ZH/')  # Root data directory.
    parser.add_argument('--result_dir', type=str, default='result/chinese_result/')
    parser.add_argument('--result_file', type=str, default="seq.results")
    parser.add_argument('--downstream_model', type=str, default='mrc')  # Evaluate Seq model on EOT if choose 'gold', evaluate Seq-Enum/MRC pipeline if 'enum'/'mrc'.
    parser.add_argument('--downstream_file', type=str, default='enumerate.results')  # Result file of downstream model: ['enumerate.results', 'bert_sqad.txt']
    parser.add_argument('--opinion_level', type=str, default='all')  # ['all', 'segment', 'sent']
    parser.add_argument('--aspect_level', type=str, default='total')  # ['subevent', 'event', 'entity', 'total']

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    Evaluate performance of Seq model.
    """
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    data_dir = args.data_dir
    result_dir = args.result_dir
    result_file = args.result_dir + args.result_file
    downstream_model = args.downstream_model
    downstream_file = args.downstream_file
    aspect_level = args.aspect_level

    if args.opinion_level == 'all' or args.opinion_level == 'segment':
        print('Segment-F1: ')
        get_segment_result()
    if args.opinion_level == 'all' or args.opinion_level == 'sent':
        print('Sent-F1: ')
        get_sent_result()