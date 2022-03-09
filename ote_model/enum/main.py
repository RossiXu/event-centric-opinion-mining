import argparse
import os
import numpy as np
import random
import torch
import transformers
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from utils import *
transformers.logging.set_verbosity_error()


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_arguments(parser):
    # Training hyper parameters
    parser.add_argument('--device', type=str, default="cuda:0",
                        choices=['cuda:0', 'cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7'],
                        help="GPU/CPU devices")
    parser.add_argument('--data_dir', type=str, default='../../data/')
    parser.add_argument('--result_dir', type=str, default='../../result/chinese_result/')
    parser.add_argument('--result_file', type=str, default='enumerate.results')
    parser.add_argument('--train_file', type=str, default="train.txt")
    parser.add_argument('--dev_file', type=str, default="dev.txt")
    parser.add_argument('--test_file', type=str, default="test.txt")
    parser.add_argument('--bert', type=str, default="bert-base-cased")
    parser.add_argument('--opinion_level', type=str, default='segment', choices=['segment', 'sent'])
    parser.add_argument('--retrain', type=int, default=1, choices=[0, 1])
    parser.add_argument('--ratio', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=8, help="default batch size is 10 (works well)")
    parser.add_argument('--num_epochs', type=int, default=10, help="Usually we set to 10.")

    # model hyperparameter
    parser.add_argument('--model_folder', type=str, default="../../model_files/", help="The name to save the model files")


    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def train_model(retrain=True, opinion_level='segment', ratio=3):
    # Init model.
    model = BertForSequenceClassification.from_pretrained(raw_model, num_labels=2)
    model.to(device)

    # Load training data.
    if retrain:
        train_df = data_preprocess(data_dir + train_file, ratio=ratio, opinion_level=opinion_level, language=language)
        dev_df = data_preprocess(data_dir + dev_file, ratio=ratio, opinion_level=opinion_level, language=language)
        eval_df = data_preprocess(data_dir + test_file, ratio=ratio, opinion_level=opinion_level, language=language)

    # Get Bert input format.
    if retrain:
        train_batches = data_batch(train_df, batch_size)

    # If model exists, evaluate and save results.
    if os.path.exists(best_model_dir) and not retrain:
        print(f"The folder " + best_model_dir + " exists. We'll use it straightly.")
        model.load_state_dict(torch.load(best_model_dir))
        return model
    elif os.path.exists(best_model_dir) and retrain:
        print(f"The folder " + best_model_dir + " exists. We'll train it and use the best.")
        model.load_state_dict(torch.load(best_model_dir))
    else:
        print(f"Begin training.")

    # Train model.
    best_acc = 0

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = train_epoches * len(train_batches)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch_i in tqdm(range(1, train_epoches+1)):
        total_train_loss = 0
        model.train()
        optimizer = lr_decay(learning_rate, lr_decay_metric, optimizer, epoch_i)
        # train model in train dataset.
        for step, batch in enumerate(tqdm(train_batches)):
            input_ids, token_type_ids, attention_mask, labels, _, _, _ = batch
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = torch.LongTensor(labels).to(device)

            model.zero_grad()

            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs.loss, outputs.logits
            total_train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        # Test model in dev dataset.
        model.eval()
        acc, _, _ = get_match_result(dev_file, model, opinion_level=opinion_level, output=False)
        if acc > best_acc:
            best_acc = acc
            print('We update best model in epoch ' + str(epoch_i))
            print('Acc in Dev dataset now is', acc, sep=' ')
            torch.save(model.state_dict(), best_model_dir)
        else:
            print('Acc does not improve.')
            print('Acc in Dev dataset now is', acc, sep=' ')
        # torch.save(model.state_dict(), model_dir + '_' + str(epoch_i))

    return model


def get_match_result(file_name, model, opinion_level='segment', output=True):
    # Init.
    eval_df = data_preprocess(data_dir + file_name, ratio=999999, opinion_level=opinion_level, language=language)
    eval_batches = data_batch(eval_df, batch_size, shuffle=False)

    # Predict.
    model.eval()
    opinions = []
    spans = []
    probs = []
    gold_aspects = {}  # opinion: gold_aspect
    with torch.no_grad():
        for step, batch in enumerate(tqdm(eval_batches)):
            input_ids, token_type_ids, attention_mask, labels, opinion, span, gold_aspect = batch

            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = torch.LongTensor(labels).to(device)

            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)

            logits = outputs.logits.detach().cpu().numpy()
            probs.extend(logits[:, 1].tolist())
            spans.extend(span)
            opinions.extend(opinion)
            gold_aspects.update(dict(zip(opinion, gold_aspect)))
    insts = []
    for o in gold_aspects.keys():
        gold_aspect = gold_aspects[o]
        o_probs = probs.copy()
        for idx, prob in enumerate(probs):
            if opinions[idx] != o:
                o_probs[idx] = -9999999
        pred_aspect = spans[o_probs.index(max(o_probs))]
        insts.append([o, gold_aspect, pred_aspect])

    correct_pred = sum([1 for inst in insts if inst[1] == inst[2]])
    total = len(insts)
    if output:
        print('Accuracy: ', correct_pred / total, correct_pred, total)

    # Save match result
    if output:
        with open(result_dir + result_file, 'w', encoding='utf-8') as f:
            for inst in insts:
                f.write("{}\t{}\t{}\n".format(inst[0], inst[1], inst[2]))

    return correct_pred / total, correct_pred, total


if __name__ == '__main__':
    """
    python main.py --device cuda:3 --lr 1e-5 --batch_size 24 --retrain 1 --bert bert-base-chinese --opinion_level segment --ratio 3
    """
    # Parse arguments.
    parser = argparse.ArgumentParser(description="Pair Classification")
    args = parse_arguments(parser)

    batch_size = args.batch_size
    train_epoches = args.num_epochs
    learning_rate = args.lr
    lr_decay_metric = args.lr_decay

    retrain = args.retrain
    opinion_level = args.opinion_level
    ratio = args.ratio
    seed = args.seed

    raw_model = args.bert
    data_dir = args.data_dir
    train_file = args.train_file
    dev_file = args.dev_file
    test_file = args.test_file
    result_dir = args.result_dir
    result_file = args.result_file
    best_model_dir = args.model_folder + 'best_enum_model_' + str(batch_size) + '_' + str(learning_rate) + '_' + str(ratio)
    model_dir = args.model_folder + 'enum_model_' + str(batch_size) + '_' + str(learning_rate) + '_' + str(ratio)

    if 'chinese' in raw_model:
        language = 'chinese'
    else:
        language = 'english'

    # Set seed
    set_seed(seed)
    # Get model
    device = args.device
    model = train_model(retrain, ratio=ratio, opinion_level=opinion_level)
    # Get result
    get_match_result(test_file, model, opinion_level)