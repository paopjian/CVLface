# Fix imgaug compatibility with numpy 2.x (np.sctypes removed)
import numpy as np
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'float': [np.float16, np.float32, np.float64],
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str, np.void],
    }

from .basic_augmenter import BasicAugmenter
from .gridsample_augmenter import GridSampleAugmenter
from .basic_augmenter_v2_numpy import BasicAugmenterV2Numpy
from .basic_augmenter_v2_cv2 import BasicAugmenterV2CV2
from .basic_augmenter_v2_gpu import BasicAugmenterV2GPU
from .gridsample_augmenter_v2_numpy import GridSampleAugmenterV2Numpy
from .gridsample_augmenter_v2_gpu import GridSampleAugmenterV2GPU
from .identity_augmenter_v2_numpy import IdentityAugmenterV2Numpy

def make_augmenter(augmentation_version, aug_params):
    if augmentation_version == 'basic':
        augmenter = BasicAugmenter(crop_augmentation_prob=aug_params.crop_augmentation_prob,
                                   photometric_augmentation_prob=aug_params.photometric_augmentation_prob,
                                   low_res_augmentation_prob=aug_params.low_res_augmentation_prob,
                                   )
    elif augmentation_version == 'gridsample':
        augmenter = GridSampleAugmenter(aug_params, input_size=112)
    elif augmentation_version == 'basic_v2_numpy':
        augmenter = BasicAugmenterV2Numpy(
            crop_augmentation_prob=aug_params.crop_augmentation_prob,
            photometric_augmentation_prob=aug_params.photometric_augmentation_prob,
            low_res_augmentation_prob=aug_params.low_res_augmentation_prob,
        )
    elif augmentation_version == 'basic_v2_cv2':
        augmenter = BasicAugmenterV2CV2(
            crop_augmentation_prob=aug_params.crop_augmentation_prob,
            photometric_augmentation_prob=aug_params.photometric_augmentation_prob,
            low_res_augmentation_prob=aug_params.low_res_augmentation_prob,
        )
    elif augmentation_version == 'basic_v2_gpu':
        augmenter = BasicAugmenterV2GPU(
            crop_augmentation_prob=aug_params.crop_augmentation_prob,
            photometric_augmentation_prob=aug_params.photometric_augmentation_prob,
            low_res_augmentation_prob=aug_params.low_res_augmentation_prob,
        )
    elif augmentation_version == 'gridsample_v2_numpy':
        augmenter = GridSampleAugmenterV2Numpy(aug_params, input_size=112)
    elif augmentation_version == 'gridsample_v2_gpu':
        augmenter = GridSampleAugmenterV2GPU(aug_params, input_size=112)
    elif augmentation_version == 'identity_v2_numpy':
        augmenter = IdentityAugmenterV2Numpy()
    else:
        raise ValueError(f'not correct augmentation version: {augmentation_version}')
    return augmenter
