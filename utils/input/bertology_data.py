# -*- coding: utf-8 -*-
# Created by li huayong on 2019/11/8
import os
import sys
import csv
import pandas


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class SequenceClassificationProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def __init__(self, class_num, is_complex=False):
        super().__init__()
        self.class_num = class_num
        self.is_complex = is_complex
        if is_complex:
            self.mode = 'complex'
        else:
            self.mode = 'simple'
        print(f'Input mode is {self.mode} !!! \n')

    def read_file(self, file_path):
        lines = []
        with open(file_path, 'r', encoding='utf-8')as f:
            for line in f:
                _lines = line.strip().split('\t')
                lines.append(_lines)
        return lines

    def read_csv(self, file_path, no_label=False):
        lines = []
        csv_data = pandas.read_csv(file_path)
        for row in csv_data.itertuples(index=False):
            sentence = row[0].replace(' ', '')
            if self.is_complex:
                history = row[1].replace(' ', '')
            else:
                history = None
            if not no_label:
                label = str(row[-1])
            else:
                label = None
            lines.append((sentence, history, label))
        return lines

    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_examples(
            # self.read_file(os.path.join(data_dir, "train.txt")),
            self.read_csv(os.path.join(data_dir, f"{self.mode}.train.csv")),
            "train",
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            # self.read_file(os.path.join(data_dir, "dev.txt")),
            self.read_csv(os.path.join(data_dir, f"{self.mode}.test.csv")),
            "dev",
        )

    def get_labels(self):
        """See base class."""
        return [str(i) for i in range(self.class_num)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples

    def get_inference_examples(self, file_path, has_label=False):
        lines = []
        if not file_path.endswith('.csv'):
            with open(file_path, 'r', encoding='utf-8')as f:
                for line in f:
                    if has_label:
                        _lines = line.strip().split('\t')[0]
                    else:
                        _lines = line.strip()
                    lines.append(_lines)
        else:
            lines = self.read_csv(file_path, no_label=not has_label)
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ('inference', i)
            if not file_path.endswith('.csv'):
                text_a = line
            else:
                text_a = line[0]
            label = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


if __name__ == '__main__':
    pass
