# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     best_result
   Description :
   Author :       Liangs
   date：          2019/7/27
-------------------------------------------------
   Change Activity:
                   2019/7/27:
-------------------------------------------------
"""


class BestResult(object):
    def __init__(self):
        self.best_f1 = -1
        self.best_step = -1
        self.best_epoch = -1
        self.best_report = ''

    def is_new_record(self, f1, global_step, epoch):
        if self.best_f1 < f1:
            self.best_f1 = f1
            self.best_step = global_step
            self.best_epoch = epoch
            return True
        else:
            return False

    def __str__(self):
        return f"BEST metrics in {self.best_step} steps, {self.best_epoch} epochs\n" \
               f"{self.best_report}"


if __name__ == '__main__':
    best_result = BestResult()
    print(best_result)
