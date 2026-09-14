from .base import BasePipeline
from models.base import BaseModel
import torch
from losses.topofr_loss import TopoFRLoss


class TrainModelClsTopoFRPipeline(BasePipeline):
    """
    Training pipeline for TopoFR: combines classification loss with topological loss
    """

    def __init__(self,
                 model: BaseModel,
                 classifier: BaseModel,
                 optimizer,
                 lr_scheduler):
        super(TrainModelClsTopoFRPipeline, self).__init__()

        self.model = model
        self.classifier = classifier
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # margin_softmax lives on the inner FC / PartialFC_V2, not on the classifier
        # wrapper, so read the flag the wrapper exposes instead of reaching inside.
        self.is_topofr = getattr(classifier, 'is_topofr', False)

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
        """
        Forward pass with TopoFR support

        For TopoFR, we need to pass:
        - input images (for topological loss on input space)
        - embeddings (for topological loss on feature space)
        """
        if len(batch) == 2:
            inputs, targets = batch
        elif len(batch) == 4:
            inputs, placeholder, targets, thetas = batch
        elif len(batch) == 7:
            inputs, targets, ldmk1, theta1, sample2, ldmk2, theta2 = batch
            if sample2.ndim != 1:
                inputs = torch.cat([inputs, sample2], dim=0)
                targets = torch.cat([targets, targets], dim=0)
        else:
            raise ValueError('not supported batch format')

        # Get features from backbone
        feats = self.model(inputs)

        # TopoFR 的拓扑损失要对齐"输入空间"和"特征空间"的拓扑结构, 所以除了 embeddings
        # 还得把原始图像传进去. embeddings 就是 feats 本身(第一个位置参数), 由
        # FCClassifier -> FC.forward 内部转发给 TopoFRLoss, 这里不用重复传.
        if self.is_topofr:
            loss = self.classifier(feats, targets.to(self.classifier.device),
                                   input_images=inputs)
        else:
            # Standard classification
            loss = self.classifier(feats, targets.to(self.classifier.device))

        return loss

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
