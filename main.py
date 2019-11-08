# -*- coding: utf-8 -*-
# Created by li huayong on 2019/11/7
import shutil
import pathlib
import torch
from datetime import datetime
from utils.arguments import parse_args
from utils.seed import set_seed
from utils.timer import Timer
from utils.input.bertology_input import load_data_for_nlu_task
from models.ClassificationTrainer import BERTologyClassificationTrainer
from models.ClassificationModel import ClassificationModel


def make_output_dir(args):
    output_dir = pathlib.Path(args.output_dir)
    assert output_dir.is_dir()
    time_str = datetime.now().strftime('_%Y-%m-%d-%H-%M-%S')
    output_dir = output_dir / (pathlib.Path(args.config_file).stem + time_str)
    if output_dir.exists():
        raise RuntimeError(f'{output_dir} exists! (maybe file or dir)')
    else:
        output_dir.mkdir()
        # 复制对应的配置文件到保存的文件夹下，保持配置和输出结果的一致
        shutil.copyfile(args.config_file, str(output_dir / pathlib.Path(args.config_file).name))
        (output_dir / 'saved_models').mkdir()
        args.output_dir = str(output_dir)
        args.dev_output_path = str(output_dir / 'dev_output.txt')
        args.dev_result_path = str(output_dir / 'dev_best_metrics.txt')
        # args.test_output_path = str(output_dir / 'test_output.txt')
        # args.test_result_path = str(output_dir / 'test_metrics.txt')
        args.save_model_dir = str(output_dir / 'saved_models')
        args.summary_dir = str(output_dir / 'summary')


def config_for_multi_gpu(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)


def load_trainer(args):
    model = ClassificationModel(args)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        print(f'Parallel Train, GPU num : {args.n_gpu}')
        args.parallel_train = True
    else:
        args.parallel_train = False
    trainer = BERTologyClassificationTrainer(args, model)
    return trainer


def main():
    args = parse_args()
    make_output_dir(args)
    config_for_multi_gpu(args)
    set_seed(args)
    with Timer('load input'):
        train_data_loader, dev_data_loader, test_data_loader = load_data_for_nlu_task(args, train=True, dev=True,
                                                                                      test=False)
    print(f'train batch size: {args.train_batch_size}')
    print(f'train data batch num: {len(train_data_loader)}')
    # 每个epoch做两次dev：
    args.eval_interval = len(train_data_loader) // 2
    print(f'eval interval: {args.eval_interval}')
    # 注意该参数影响学习率warm up
    args.max_train_steps = len(train_data_loader) * args.max_train_epochs
    print(f'max steps: {args.max_train_steps}')
    if not args.early_stop:
        print(f'do not use early stop, training will last {args.max_train_epochs} epochs')

    with Timer('load trainer'):
        trainer = load_trainer(args)
    with Timer('Train'):
        trainer.train(train_data_loader, dev_data_loader)


if __name__ == '__main__':
    main()
