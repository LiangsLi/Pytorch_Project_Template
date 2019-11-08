# -*- coding: utf-8 -*-
# Created by li huayong on 2019/11/7
import os
import pathlib

import torch
import torch.nn as nn
from module.bertology_encoder import BERTologyEncoder


class BaseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.encoder in ['bert', 'xlnet', 'xlm', 'roberta']:
            self.encoder = BERTologyEncoder(bertology_type=args.bertology_type,
                                            bertology_path=args.saved_model_path,
                                            bertology_output=args.bertology_output,
                                            )
        else:
            raise ValueError(f'Unsupported Encoder Type {args.encoder}')
        self.classifier = nn.Sequential(
            nn.Linear(args.encoder_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, args.class_num),
        )
        self.classifier.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, inputs):
        assert isinstance(inputs, dict)
        encoder_output = self.encoder(**inputs)
        return self.classifier(encoder_output)

    def save_pretrained(self, save_directory, weight_file_name="pytorch_model.bin"):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(
            save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, weight_file_name)
        torch.save(model_to_save.state_dict(), output_model_file)
        print("Model weights saved in {}".format(output_model_file))

    @classmethod
    def from_pretrained(cls, args, saved_model_path=None, weight_file_name="pytorch_model.bin",
                        initialize_from_bertology=False):
        """
            注意，这里不支持训练BERT的LM和Next Sentence任务
        :param args:
        :param saved_model_path:
        :param weight_file_name:
        :param initialize_from_bertology:
        :return:
        """
        import re
        from collections import OrderedDict

        if saved_model_path:
            args.saved_model_path = saved_model_path
        model = cls(args)
        resolved_archive_file = pathlib.Path(args.saved_model_path) / weight_file_name
        assert resolved_archive_file.exists()
        state_dict = torch.load(str(resolved_archive_file), map_location='cpu')
        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        # 如果模型从BERTology预训练模型加载，则需要修改参数的名称以匹配现有的模型架构
        if initialize_from_bertology:
            rename_state_dict = OrderedDict()
            for key in state_dict.keys():
                # 这里 ^bert 表示以bert开头的模型参数，目前仅仅支持BERT类型的模型，XLNET等需要补充
                rename_key = re.sub('^bert.', 'encoder.bertology.', key)
                rename_state_dict[rename_key] = state_dict[key]
            state_dict = rename_state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model)

        if len(missing_keys) > 0:
            print("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            print("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                model.__class__.__name__, "\n\t".join(error_msgs)))

        model.eval()

        return model


if __name__ == '__main__':
    pass
