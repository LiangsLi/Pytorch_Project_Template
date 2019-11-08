# -*- coding: utf-8 -*-
# Created by li huayong on 2019/11/7
import pathlib

import torch
import torch.nn as nn
from pytorch_transformers import (BertConfig,
                                  BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer, BertModel)

from utils.information import debug_print
from module.layer_attention import LayerAttention
from module.transformer_layer import TransformerSentenceEncoderLayer

BERTology_MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    # 'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    # 'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    # 'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


class BERTologyEncoder(nn.Module):
    def __init__(self,
                 no_cuda=False,
                 bertology_type='bert',
                 bertology_path=None,
                 bertology_output='last_four_sum',
                 layer_attention_dropout=0.2,
                 after_layer='none',
                 after_layer_num=0,
                 after_layer_dropout=0.2,
                 encoder_dropout=0.2,
                 use_fine_tuned_model=False
                 ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        self.n_gpu = torch.cuda.device_count()

        self.bertology_type = bertology_type.lower()
        self.bertology_path = bertology_path
        self.bertology_output = bertology_output
        self.bertology_config_class, self.bertology_model_class, _ = BERTology_MODEL_CLASSES[self.bertology_type]

        self.bertology_config = self.bertology_config_class.from_pretrained(self.bertology_path)
        self.bertology_config.output_hidden_states = True
        # 注意这里不加载BERT的预训练参数
        # BERT的参数通过Model.from_pretrained方法加载
        self.bertology = self.bertology_model_class(config=self.bertology_config)
        self.encoder_dropout = nn.Dropout(encoder_dropout)
        if self.bertology_output == 'attention':
            self.layer_attention = LayerAttention(self.bertology_config.num_hidden_layers, do_layer_norm=False,
                                                  dropout=layer_attention_dropout)
        else:
            self.layer_attention = None
        if after_layer == 'transformer':
            self.after_encoder = nn.ModuleList(
                [
                    TransformerSentenceEncoderLayer(embedding_dim=self.bertology_config.hidden_size,
                                                    dropout=after_layer_dropout)
                    for _ in range(after_layer_num)
                ]
            )
        else:
            self.after_encoder = None

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        bertology_output = self.bertology(input_ids, token_type_ids, attention_mask)
        last_layer_hidden_states = bertology_output[0]
        last_layer_cls = last_layer_hidden_states[:, 0, :]
        all_layer_hidden_states = bertology_output[2][1:]
        all_layers_cls = [states[:, 0, :] for states in all_layer_hidden_states]
        if self.after_encoder is None:
            _last = last_layer_cls
            _all = all_layers_cls
        else:
            _last = last_layer_hidden_states
            _all = all_layer_hidden_states
        if self.bertology_output == 'last':
            encoder_output = _last
        elif self.bertology_output == 'last_four_sum':
            last_four_hidden_states = torch.stack(_all[-4:])
            encoder_output = torch.sum(last_four_hidden_states, 0)
        elif self.bertology_output == 'attention':
            encoder_output = self.layer_attention(_all)
        else:
            raise Exception('bad bert output mode')

        if self.after_encoder is not None:
            # attention_mask has 1 for real tokens and 0 for padding tokens.
            attention_pad_mask = torch.eq(attention_mask, 0)
            # 确保pad位置为0
            encoder_output *= (1 - attention_pad_mask.unsqueeze(-1).type_as(encoder_output))

            # batch X Seq_len X dim -> Seq_len X batch X dim
            encoder_output = encoder_output.transpose(0, 1)
            for layer in self.after_encoder:
                encoder_output, _ = layer(encoder_output, self_attn_padding_mask=attention_mask)
            # Seq_len X batch X dim -> batch X Seq_len X dim
            encoder_output = encoder_output.transpose(0, 1)
            # 取出CLS对应的表示作为句子级别的表示
            encoder_output = encoder_output[:, 0, :]
        return self.encoder_dropout(encoder_output)


if __name__ == '__main__':
    pass
