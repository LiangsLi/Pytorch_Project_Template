# -*- coding: utf-8 -*-
# Created by li huayong on 2019/11/7
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from abc import ABCMeta, abstractmethod, ABC
from utils.model.get_optimizer import get_optimizer
from utils.best_result import BestResult
from utils.seed import set_seed
from utils.model.label_smoothing import label_smoothed_kl_div_loss
from sklearn.metrics import f1_score, classification_report

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, args, model=None):
        self.model = model
        self.args = args
        if self.model is not None:
            self.optimizer, self.optim_scheduler = get_optimizer(args, model)
            # 这里用交叉熵为例子：
            # 也可以在forward中定义损失函数
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.optimizer = self.optim_scheduler = None
            self.loss_func = None

    def check_model(func):
        """
            类内部装饰器，用来检查类方法的参数
        :return:
        """
        def wrapper(self, *args, **kwargs):
            if self.model is None:
                raise RuntimeError('The model is not initialized, please load the model first.')
            return func(self, *args, **kwargs)

        return wrapper

    @abstractmethod
    def _unpack_batch(self, args, batch):
        raise NotImplementedError('NotImplemented')

    def _update_and_predict(self, logits, y_trues=None, calc_loss=True, update=True, calc_prediction=False):
        if calc_loss:
            assert y_trues is not None
            loss = self.loss_func(logits.view(-1, self.args.class_num), y_trues.view(-1))
            if self.args.n_gpu > 1:
                loss = loss.mean()
            if update:
                loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                if self.optim_scheduler:
                    self.optim_scheduler.step()
                self.optimizer.step()
                self.model.zero_grad()
                self.optimizer.zero_grad()
            loss = loss.detach().cpu().item()
        else:
            loss = None

        if calc_prediction:
            batch_prediction = np.argmax(logits.detach().cpu().numpy(), axis=1)
        else:
            batch_prediction = None
        return loss, batch_prediction

    @check_model
    def train(self, train_data_loader, dev_data_loader=None):
        best_result = BestResult()
        self.model.zero_grad()
        set_seed(self.args)
        train_stop = False
        summary_writer = SummaryWriter(log_dir=self.args.summary_dir)
        global_step = 0
        for epoch in range(self.args.max_train_epochs + 1):
            epoch_train_loss = 0
            train_data_loader = tqdm(train_data_loader, desc=f'Training epoch {epoch}')
            for step, batch in enumerate(train_data_loader):
                batch = tuple(t.to(self.args.device) for t in batch)
                self.model.train()
                inputs, y_trues = self._unpack_batch(self.args, batch)
                logits = self.model(inputs)
                loss, _ = self._update_and_predict(logits, y_trues, calc_loss=True, update=True, calc_prediction=False)
                global_step += 1
                if loss is not None:
                    epoch_train_loss += loss
                if global_step % self.args.eval_interval == 0:
                    summary_writer.add_scalar('loss/train', loss, global_step)
                    if dev_data_loader:
                        f1, report = self.dev(dev_data_loader)
                        summary_writer.add_scalar('metrics/f1', f1, global_step)
                        if best_result.is_new_record(f1, global_step, epoch):
                            best_result.best_report = report
                            print(f"\n## NEW BEST RESULT in epoch {epoch} ##")
                            print(best_result)
                if self.args.early_stop and (epoch - best_result.best_epoch) > self.args.early_stop_epoch:
                    print(f'\n## Early stop in epoch:{epoch} ##')
                    train_stop = True
                    break
            if train_stop:
                break
            summary_writer.add_scalar('epoch_average_loss', epoch_train_loss / len(train_data_loader), epoch)
        with open(self.args.dev_result_path, 'w', encoding='utf-8')as f:
            f.write(str(best_result) + '\n')
        print("\n## BEST RESULT in Training ##")
        print(best_result)
        summary_writer.close()
        print('train stop')

    @check_model
    def dev(self, dev_data_loader):
        dev_data_loader = tqdm(dev_data_loader, desc='Eval')
        predictions = []
        all_y_tures = []
        for step, batch in enumerate(dev_data_loader):
            self.model.eval()
            batch = tuple(t.to(self.args.device) for t in batch)
            inputs, y_trues = self._unpack_batch(self.args, batch)
            logits = self.model(inputs)
            _, batch_prediction = self._update_and_predict(logits, y_trues=None,
                                                           calc_loss=False, calc_prediction=True, update=False)
            predictions += batch_prediction.tolist()
            all_y_tures += y_trues.detach().cpu().numpy().tolist()
        # 这里用F1作为评价指标
        f1 = f1_score(y_true=all_y_tures, y_pred=predictions, average='macro')
        report = classification_report(y_true=all_y_tures, y_pred=predictions)
        return f1, report

    @check_model
    def inference(self):
        raise NotImplementedError('')


class BERTologyClassificationTrainer(BaseTrainer, ABC):
    """
        BaseTrainer子类,必须实现 _unpack_batch
        这里用BERT文本分类举例
    """
    def _unpack_batch(self, args, batch):
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
        }
        y_trues = batch[3]
        return inputs, y_trues


if __name__ == '__main__':
    pass
