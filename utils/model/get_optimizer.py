# -*- coding: utf-8 -*-
# Created by li huayong on 2019/10/7
import math

import torch
import pytorch_transformers.optimization as huggingfaceOptim  # 避免和torch.optim重名
from utils.information import debug_print


def get_optimizer(args, model):
    args.warmup_steps = math.ceil(args.warmup_prop * args.max_train_steps)
    if args.optimizer == 'adamw-bert':
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = huggingfaceOptim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
                                           betas=(args.beta1, args.beta2))
        scheduler = huggingfaceOptim.WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps,
                                                          t_total=args.max_train_steps)
        debug_print('\n - Use Huggingface\'s AdamW Optimizer')
    elif args.optimizer == 'adamw-torch':
        try:
            from torch.optim import AdamW
        except ImportError as e:
            debug_print(f'torch version: {torch.__version__}')
            raise e
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
                          betas=(args.beta1, args.beta2))
        scheduler = huggingfaceOptim.WarmupLinearSchedule(optimizer,
                                                          warmup_steps=args.warmup_steps,
                                                          t_total=args.max_train_steps)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        scheduler = None
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)
        scheduler = None
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=args.betas, eps=args.eps,
                                     weight_decay=args.weight_decay)
        scheduler = None
    elif args.rnn_optimizer == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters())  # use default lr
        scheduler = None
    else:
        raise Exception("Unsupported optimizer: {}".format(args.optimizer))
    return optimizer, scheduler


if __name__ == '__main__':
    pass
