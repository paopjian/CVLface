from .margin_loss import CombinedMarginLoss, ArcFace, CosFace
from .adaface import AdaFaceLoss
from .topofr_loss import TopoFRLoss

def get_margin_loss(loss_config):
    if loss_config.margin_loss_name == 'margin':
        margin_loss = CombinedMarginLoss(
            64,
            loss_config.margin_list[0],
            loss_config.margin_list[1],
            loss_config.margin_list[2],
            loss_config.interclass_filtering_threshold
        )
    elif loss_config.margin_loss_name == 'adaface':
        margin_loss = AdaFaceLoss(
            64,
            m=loss_config.m,
            h=loss_config.h,
            t_alpha=loss_config.t_alpha,
            interclass_filtering_threshold=loss_config.interclass_filtering_threshold
        )
    elif loss_config.margin_loss_name == 'arcface':
        s = getattr(loss_config, 's', 64.0)
        m = getattr(loss_config, 'm', 0.5)
        margin_loss = ArcFace(s=s, margin=m)
    elif loss_config.margin_loss_name == 'cosface':
        s = getattr(loss_config, 's', 64.0)
        m = getattr(loss_config, 'm', 0.4)
        margin_loss = CosFace(s=s, m=m)
    elif loss_config.margin_loss_name == 'topofr':
        # TopoFR: wraps a base margin loss with topological loss
        base_loss_name = getattr(loss_config, 'base_loss', 'arcface')

        # Create base margin loss
        if base_loss_name == 'arcface':
            s = getattr(loss_config, 's', 64.0)
            m = getattr(loss_config, 'm', 0.5)
            base_margin_loss = ArcFace(s=s, margin=m)
        elif base_loss_name == 'cosface':
            s = getattr(loss_config, 's', 64.0)
            m = getattr(loss_config, 'm', 0.4)
            base_margin_loss = CosFace(s=s, m=m)
        elif base_loss_name == 'adaface':
            base_margin_loss = AdaFaceLoss(
                64,
                m=loss_config.m,
                h=loss_config.h,
                t_alpha=loss_config.t_alpha,
                interclass_filtering_threshold=loss_config.interclass_filtering_threshold
            )
        else:
            raise ValueError(f"Not supported base loss for TopoFR: {base_loss_name}")

        # Wrap with TopoFR loss
        topo_weight = getattr(loss_config, 'topo_weight', 0.1)
        use_gum = getattr(loss_config, 'use_gum', True)
        temp = getattr(loss_config, 'temp', 1)

        margin_loss = TopoFRLoss(
            margin_loss=base_margin_loss,
            topo_weight=topo_weight,
            use_gum=use_gum,
            temp=temp
        )
    elif loss_config.margin_loss_name == 'none':
        margin_loss = None
    else:
        raise ValueError("Not implemented loss margin_loss_name")
    return margin_loss

