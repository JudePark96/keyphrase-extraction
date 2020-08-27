__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import argparse
import logging
import os
import random
import sys
import torch.nn as nn
import numpy as np
import torch

from io import open
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from tqdm import tqdm, trange

from transformers import BertTokenizer, BertModel

from dataset.kp20k_dataset import KP20KTrainingDataset
from models.span_extraction_model import SpanClassifier
from utils.optimization import AdamW, WarmupLinearSchedule
from torch.utils.tensorboard import SummaryWriter


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir", default='output', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument('--log_dir', default='./runs', type=str)
    parser.add_argument('--bert_model_config', default='bert-base-uncased', type=str)
    parser.add_argument('--model_type', default='baseline', type=str)
    parser.add_argument('--checkpoint', type=str)

    # Other parameters
    parser.add_argument("--train_file", default='./rsc/features/kp20k.feature.train.256.32.hdf5', type=str,
                        help="preprocessed training file")
    parser.add_argument("--valid_file", default='./rsc/features/kp20k.feature.valid.256.32.hdf5', type=str,
                        help="preprocessed validation file")
    parser.add_argument("--test_file", default='./rsc/features/kp20k.feature.test.256.32.hdf5', type=str,
                        help="preprocessed test file")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--valid_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--test_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--valid_per_step", default=3000, type=int, help="Validation per step.")
    parser.add_argument("--num_workers", default=16, type=int, help="number of workers option.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=4.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(
        device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    output_dir = args.output_dir + args.model_type + '/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bert_model = BertModel.from_pretrained(args.bert_model_config)

    model = SpanClassifier(bert_model, args.model_type)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))

    if n_gpu > 1:
        model = nn.DataParallel(model)

    num_params = count_parameters(model)
    logger.info("Total Parameter: %d" % num_params)
    model.to(device)

    print(model)

    train_examples = KP20KTrainingDataset(args.train_file)

    valid_flag, test_flag = False, False

    if args.valid_file:
        valid_flag = True
        valid_examples = KP20KTrainingDataset(args.valid_file)
        valid_sampler = RandomSampler(valid_examples)
        valid_dataloader = DataLoader(valid_examples, sampler=valid_sampler, batch_size=args.valid_batch_size)

    if args.test_file:
        test_flag = True
        test_examples = KP20KTrainingDataset(args.test_file)
        test_sampler = RandomSampler(test_examples)
        test_dataloader = DataLoader(test_examples, sampler=test_sampler, batch_size=args.test_batch_size)

    num_train_optimization_steps = int(len(train_examples) / args.train_batch_size) * args.num_train_epochs

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=num_train_optimization_steps * 0.1,
                                     t_total=num_train_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("  Num orig examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    num_train_step = num_train_optimization_steps

    train_sampler = RandomSampler(train_examples)
    # TODO => Add argument num_worker option
    train_dataloader = DataLoader(train_examples, sampler=train_sampler, batch_size=args.train_batch_size,
                                  num_workers=args.num_workers)

    logger.info('Start training ...')
    summary_writer = SummaryWriter(args.log_dir)

    model.train()
    global_step = 0
    epoch = 0
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        iter_bar = tqdm(train_dataloader, desc="Train(XX Epoch) Step(XX/XX) (Mean loss=X.X) (loss=X.X)")
        tr_step, total_loss, mean_loss = 0, 0., 0.
        for step, batch in enumerate(iter_bar):
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self

            if args.model_type == 'baseline':
                loss, s_loss, e_loss = model(batch)
                if n_gpu > 1:
                    s_loss, e_loss = s_loss.mean(), e_loss.mean()
            elif args.model_type == 'span_rank':
                loss, s_loss, e_loss, s_rank_loss, e_rank_loss = model(batch)
                if n_gpu > 1:
                    s_loss, e_loss, s_rank_loss, e_rank_loss = s_loss.mean(), \
                                                               e_loss.mean(), \
                                                               s_rank_loss.mean(), \
                                                               e_rank_loss.mean()
            else:
                loss, s_loss, e_loss = model(batch)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            tr_step += 1
            total_loss += loss
            mean_loss = total_loss / tr_step
            iter_bar.set_description("Train Step(%d / %d) (Mean loss=%5.5f) (loss=%5.5f)" %
                                     (global_step, num_train_step, mean_loss, loss.item()))

            if global_step % args.valid_per_step == 0 and valid_flag:
                model.eval()
                for valid_batch in valid_dataloader:
                    with torch.no_grad():
                        valid_result = model(valid_batch, is_eval=True)
                        print(valid_result['s_score'].shape)
                        print(valid_result['s_gt'].shape)
                    pass
                model.train()

            if global_step % 100 == 0:
                summary_writer.add_scalar('Train/Mean_Loss', mean_loss, global_step)
                summary_writer.add_scalar('Train/Loss', loss.item(), global_step)
                summary_writer.add_scalar('Train/s_loss', s_loss.item(), global_step)
                summary_writer.add_scalar('Train/e_loss', e_loss.item(), global_step)

                if args.model_type == 'span_rank':
                    summary_writer.add_scalar('Train/s_rank_loss', s_rank_loss.item(), global_step)
                    summary_writer.add_scalar('Train/e_rank_loss', e_rank_loss.item(), global_step)

        logger.info("** ** * Saving file * ** **")
        model_checkpoint = "span_%d.bin" % (epoch)
        logger.info(model_checkpoint)
        output_model_file = os.path.join(output_dir, model_checkpoint)

        if n_gpu > 1:
            torch.save(model.module.state_dict(), output_model_file)
        else:
            torch.save(model.state_dict(), output_model_file)
        epoch += 1


if __name__ == "__main__":
    main()