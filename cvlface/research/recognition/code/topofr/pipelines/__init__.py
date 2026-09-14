from omegaconf import DictConfig
from typing import Union, Any
from .train_model_cls_pipeline import TrainModelClsPipeline
from .train_model_cls_topofr_pipeline import TrainModelClsTopoFRPipeline
from .train_keypoint_model_cls_pipeline import TrainKeypointModelClsPipeline
from .infer_model_pipeline import InferModelPipeline
from .infer_aligner_model_pipeline import InferAlignerModelPipeline
from .infer_aligner_keypoint_model_pipeline import InferAlignerKeypointModelPipeline
from .infer_aligner_keypoint_model_nmescore_pipeline import InferAlignerKeypointModelNMEScorePipeline

def pipeline_from_config(pipeline_config: Union[DictConfig, dict],
                         model: Any=None,
                         classifier: Any=None,
                         aligner: Any=None,
                         optimizer: Any=None,
                         lr_scheduler: Any=None):
    # NOTE: pipeline_config is cfg.pipelines only, so it cannot inspect cfg.losses.
    # Select the TopoFR pipeline explicitly via pipelines=configs/train_model_cls_topofr

    if pipeline_config.name == 'TrainModelClsPipeline':
       pipeline = TrainModelClsPipeline(model, classifier, optimizer, lr_scheduler)
    elif pipeline_config.name == 'TrainModelClsTopoFRPipeline':
       pipeline = TrainModelClsTopoFRPipeline(model, classifier, optimizer, lr_scheduler)
    elif pipeline_config.name == 'TrainKeypointModelClsPipeline':
       pipeline = TrainKeypointModelClsPipeline(model, classifier, optimizer, lr_scheduler)
    else:
        raise NotImplementedError(f"pipeline {pipeline_config.name} not implemented")

    if pipeline_config.resume:
        epoch, step, n_images_seen = pipeline.resume_from_dir(pipeline_config.resume)
        start_epoch = epoch + 1

    else:
        start_epoch, step, n_images_seen = 0, 0, 0
    pipeline.start_epoch = start_epoch
    pipeline.step = step
    pipeline.n_images_seen = n_images_seen

    return pipeline

def pipeline_from_name(name: str, model: Any=None, aligner: Any=None):
    if name == 'infer_model_pipeline':
        pipeline = InferModelPipeline(model)
    elif name == 'infer_aligner_model_pipeline':
        assert aligner.has_params()
        pipeline = InferAlignerModelPipeline(aligner=aligner, model=model)
    elif name == 'infer_aligner_keypoint_model_pipeline':
        assert aligner.has_params()
        pipeline = InferAlignerKeypointModelPipeline(aligner=aligner, model=model)
    elif name == 'infer_aligner_keypoint_model_nmescore_pipeline':
        assert aligner.has_params()
        pipeline = InferAlignerKeypointModelNMEScorePipeline(aligner=aligner, model=model)
    else:
        raise NotImplementedError(f"pipeline {name} not implemented")

    return pipeline