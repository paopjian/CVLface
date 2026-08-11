from ..base import BaseModel
from .model import IR_101, IR_50, IR_18
from torchvision import transforms


class IResNetModel(BaseModel):

    """
    A class representing a model for IResNet architectures. It supports creating
    models with specific configurations such as IR_50 and IR_101.

    Attributes:
        net (torch.nn.Module): The IResNet network (either IR_50 or IR_101).
        config (object): The configuration object with model specifications.
    """


    def __init__(self, net, config):
        super(IResNetModel, self).__init__(config)
        self.net = net
        self.config = config


    @classmethod
    def from_config(cls, config):
        if config.name == 'ir50':
            net = IR_50(input_size=(112,112), output_dim=config.output_dim,
                        dropout=getattr(config, 'dropout', 0.4))
        elif config.name == 'ir101':
            net = IR_101(input_size=(112,112), output_dim=config.output_dim,
                         dropout=getattr(config, 'dropout', 0.4))
        elif config.name == 'ir18':
            net = IR_18(input_size=(112,112), output_dim=config.output_dim,
                        dropout=getattr(config, 'dropout', 0.4))
        else:
            raise NotImplementedError

        model = cls(net, config)
        model.eval()
        return model

    def forward(self, x, coreface=False, dropout=None, clean_route=False):
        if self.input_color_flip:
            x = x.flip(1)
        if coreface:
            if clean_route:
                return self.net.forward_triple(x, dropout=dropout)
            return self.net.forward_double(x, dropout=dropout)
        return self.net(x)

    def forward_coreface(self, x, dropout=None):
        if self.input_color_flip:
            x = x.flip(1)
        return self.net.forward_double(x, dropout=dropout)

    def set_dropout(self, probability):
        self.net.set_dropout(probability)

    def make_train_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return transform

    def make_test_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return transform

def load_model(model_config):
    model = IResNetModel.from_config(model_config)
    return model
