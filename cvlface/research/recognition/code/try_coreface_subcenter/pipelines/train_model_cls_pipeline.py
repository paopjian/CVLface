from .base import BasePipeline
from models.base import BaseModel
from losses import ContraFaceLoss
import torch

class TrainModelClsPipeline(BasePipeline):

    def __init__(self,
                 model:BaseModel,
                 classifier:BaseModel,
                 optimizer,
                 lr_scheduler,
                 pipeline_config=None):
        super(TrainModelClsPipeline, self).__init__()

        self.model = model
        self.classifier = classifier
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        pipeline_config = pipeline_config or {}
        self.coreface_enabled = bool(getattr(pipeline_config, 'coreface_enabled', False))
        self.coreface_start_epoch = int(getattr(pipeline_config, 'coreface_start_epoch', 8))
        self.coreface_dropout = float(getattr(pipeline_config, 'coreface_dropout', 0.4))
        self.coreface_dropout2 = float(getattr(pipeline_config, 'coreface_dropout2', 0.6))
        self.coreface_weight1 = float(getattr(pipeline_config, 'coreface_weight1', 0.5))
        self.coreface_weight2 = float(getattr(pipeline_config, 'coreface_weight2', 0.5))
        self.coreface_weight_contrast = float(getattr(pipeline_config, 'coreface_weight_contrast', 0.05))
        self.coreface_weight_contrast_reverse = float(
            getattr(pipeline_config, 'coreface_weight_contrast_reverse', 0.0)
        )
        self.coreface_loss = ContraFaceLoss()
        self.coreface_active = False
        self.last_losses = {}

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
        targets = targets.to(self.classifier.device)
        if self.coreface_active and self.model.has_trainable_params():
            feat1, feat2 = self.model(inputs, coreface=True, dropout=self.coreface_dropout2)
            loss1 = self.classifier(feat1, targets.clone())
            loss2 = self.classifier(feat2, targets.clone())
            contrast = self.coreface_loss(feat1, feat2, targets)
            contrast_reverse = self.coreface_loss(feat2, feat1, targets)
            loss = (
                self.coreface_weight1 * loss1
                + self.coreface_weight2 * loss2
                + self.coreface_weight_contrast * contrast
                + self.coreface_weight_contrast_reverse * contrast_reverse
            )
            self.last_losses = {
                'train/coreface_loss_view1': loss1.detach(),
                'train/coreface_loss_view2': loss2.detach(),
                'train/coreface_loss_contrast': contrast.detach(),
                'train/coreface_loss_contrast_reverse': contrast_reverse.detach(),
            }
            return loss

        feats = self.model(inputs)
        loss = self.classifier(feats, targets.clone())
        self.last_losses = {'train/coreface_loss_view1': loss.detach()}
        return loss

    def set_epoch(self, epoch):
        self.coreface_active = self.coreface_enabled and epoch >= self.coreface_start_epoch
        if hasattr(self.model, 'set_dropout'):
            self.model.set_dropout(self.coreface_dropout2 if self.coreface_active else self.coreface_dropout)
        else:
            model = getattr(self.model, 'module', self.model)
            if hasattr(model, 'set_dropout'):
                model.set_dropout(self.coreface_dropout2 if self.coreface_active else self.coreface_dropout)


    def train(self):
        if not self.model.config.freeze:
            self.model.train()
        else:
            self.model.eval()
            # 只对解冻范围内的 BN 恢复 train mode，让其 running_stats 适应新数据
            for name, m in self.model.named_modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
                        m.train()
        if not self.classifier.config.freeze:
            self.classifier.train()


    def eval(self):
        self.model.eval()
        self.classifier.eval()
