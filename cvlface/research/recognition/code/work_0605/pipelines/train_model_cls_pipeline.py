from .base import BasePipeline
from models.base import BaseModel
import torch

class TrainModelClsPipeline(BasePipeline):

    def __init__(self,
                 model:BaseModel,
                 classifier:BaseModel,
                 optimizer,
                 lr_scheduler):
        super(TrainModelClsPipeline, self).__init__()

        self.model = model
        self.classifier = classifier
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    @property
    def module_names_list(self):
        return ['model', 'classifier', 'optimizer', 'lr_scheduler']

    def integrity_check(self, dataset):
        # color space check
        dataset_color_space = dataset.color_space
        assert dataset_color_space == self.model.config.color_space
        self.color_space = dataset_color_space
        self.make_train_transform()

    def make_train_transform(self):
        return self.model.make_train_transform()

    def __call__(self, batch):
        if len(batch) == 2:
            inputs, targets  = batch
        elif len(batch) == 4:
            inputs, placeholder, targets, thetas = batch
        elif len(batch) == 7:
            inputs, targets, ldmk1, theta1, sample2, ldmk2, theta2 = batch
            if sample2.ndim != 1:
                inputs = torch.cat([inputs, sample2], dim=0)
                targets = torch.cat([targets, targets], dim=0)

        else:
            raise ValueError('not supported batch format')
        feats = self.model(inputs)
        loss = self.classifier(feats, targets.to(self.classifier.device))
        return loss


    def train(self):
        # 先将整个 model 置为 eval，防止冻结层的 BN running stats 被污染。
        # peft(freeze/part_freeze)通过 requires_grad=False 冻结参数，但不改
        # model.config.freeze，因此不能依赖 config.freeze 来决定 train/eval 模式。
        self.model.eval()
        # 再只对 requires_grad=True 的 BN 层恢复 train mode，让其 running_stats
        # 随新数据更新（全量训练时所有 BN 都会恢复，部分冻结时只有解冻层恢复）。
        for name, m in self.model.named_modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
                    m.train()
        if not self.classifier.config.freeze:
            self.classifier.train()


    def eval(self):
        self.model.eval()
        self.classifier.eval()


