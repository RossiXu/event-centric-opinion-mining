if __name__ == '__main__':
    """
    Split raw data files by split ratio.
    """
    root_data_dir = '../data/'
    raw_data_file = 'english_data.txt'
    train_file_name = 'train.txt'
    test_file_name = 'test.txt'
    dev_file_name = 'dev.txt'
    split_ratio = [7, 1, 2]

    with open(root_data_dir + raw_data_file, 'r', encoding='utf-8') as f:
        lines = f.read()
    psgs = lines.split('-'*54)
    psgs = [psg for psg in psgs if len(psg.strip())]

    split_ratio = [ratio / sum(split_ratio) for ratio in split_ratio]
    total_num = len(psgs)
    split_index = [int(split_ratio[0]*total_num), int(sum(split_ratio[0:2])*total_num)]
    trains = psgs[:split_index[0]]
    devs = psgs[split_index[0]:split_index[1]]
    tests = psgs[split_index[1]:total_num]
    print(len(psgs))
    with open(root_data_dir + train_file_name, 'w', encoding='utf-8') as f:
        data = ('-'*54).join(trains).strip()
        print('Doc Num of train dataset: ', len(trains))
        f.write(data + '\n')
        f.write('-'*54 + '\n')
    with open(root_data_dir + dev_file_name, 'w', encoding='utf-8') as f:
        data = ('-' * 54).join(devs).strip()
        print('Doc number of dev dataset: ', len(devs))
        f.write(data + '\n')
        f.write('-' * 54 + '\n')
    with open(root_data_dir + test_file_name, 'w', encoding='utf-8') as f:
        data = ('-' * 54).join(tests).strip()
        print('Doc number of test dataset: ', len(tests))
        f.write(data + '\n')
        f.write('-' * 54 + '\n')