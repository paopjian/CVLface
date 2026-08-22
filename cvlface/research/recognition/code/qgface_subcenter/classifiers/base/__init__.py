import os
from typing import Union
import torch
from torch import device
from .utils import get_parameter_device, get_parameter_dtype, save_state_dict_and_config, load_state_dict_from_path
from general_utils.os_utils import natural_sort

class BaseClassifier(torch.nn.Module):

    def __init__(self, config=None):
        super(BaseClassifier, self).__init__()
        self.config = config

    @classmethod
    def from_config(cls, classifier_cfg, margin_loss_fn, model_cfg, dataset_cfg, rank, world_size) -> "BaseClassifier":
        raise NotImplementedError('from_config must be implemented in subclass')

    def forward(self, local_embeddings, local_labels):
        raise NotImplementedError('from_config must be implemented in subclass')


    @property
    def device(self) -> device:
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        return get_parameter_dtype(self)

    def num_parameters(self, only_trainable: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)

    def has_trainable_params(self):
        for param in self.parameters():
            if param.requires_grad:
                return True
        return False

    def save_pretrained(
        self,
        save_dir: Union[str, os.PathLike],
        name: str = 'model.pt',
        rank: int = 0,
    ):
        rank_added_name = os.path.splitext(name)[0] + f'_rank{rank}' + os.path.splitext(name)[1]
        save_path = os.path.join(save_dir, rank_added_name)
        save_state_dict_and_config(self.state_dict(), self.config, save_path)


    def load_state_dict_from_path(self, pretrained_model_path):
        save_dir = pretrained_model_path
        all_partitions = [name for name in os.listdir(save_dir) if 'classifier_rank' in name and name.endswith('.pt')]
        all_partitions = natural_sort(all_partitions)
        ckpt_worldsize = len(all_partitions)

        if self.world_size == ckpt_worldsize:
            rank_file = os.path.join(save_dir, f'classifier_rank{self.rank}.pt')
            state_dict = load_state_dict_from_path(rank_file)
            result = self.load_state_dict(state_dict, strict=False)
            print(f'加载预训练模型 rank{self.rank} (same world_size={ckpt_worldsize}):', rank_file)
            print(result)
        else:
            print(f'Redistributing classifier: ckpt_worldsize={ckpt_worldsize} -> current_worldsize={self.world_size}')
            part_ckpts = [torch.load(os.path.join(save_dir, name), map_location='cpu') for name in all_partitions]
            combined_weight = torch.cat([ckpt['partial_fc.weight'] for ckpt in part_ckpts], dim=0)
            num_subcenters = self.partial_fc.num_subcenters
            total_ckpt_num_subjects = combined_weight.shape[0] // num_subcenters
            print(f'Combined classifier weight: {total_ckpt_num_subjects} classes from {ckpt_worldsize} ranks')

            state_dict = part_ckpts[0]
            row_start = self.partial_fc.class_start * num_subcenters
            num_rows = self.partial_fc.num_local * num_subcenters
            sub_center = combined_weight[row_start:row_start + num_rows, :]
            if sub_center.shape[0] < num_rows:
                extra = torch.zeros(num_rows - sub_center.shape[0], sub_center.shape[1])
                sub_center = torch.cat([sub_center, extra], dim=0)
            state_dict['partial_fc.weight'] = sub_center

            if 'partial_fc.batch_mean' in part_ckpts[0]:
                state_dict['partial_fc.batch_mean'] = part_ckpts[0]['partial_fc.batch_mean']
            if 'partial_fc.batch_std' in part_ckpts[0]:
                state_dict['partial_fc.batch_std'] = part_ckpts[0]['partial_fc.batch_std']

            result = self.load_state_dict(state_dict, strict=False)
            print(f'加载预训练模型 rank{self.rank} (redistributed {ckpt_worldsize}->{self.world_size}):', save_dir)
            print(result)
