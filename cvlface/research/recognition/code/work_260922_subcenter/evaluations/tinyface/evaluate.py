import os
import time
import numpy as np
import torch
from .metrics import DIR_FAR, closed_set_cmc, get_tinyface_num_threads


def evaluate(
        all_features,
        image_paths,
        meta,
        ranks=[1, 5, 20]
):
    stage_start = time.perf_counter()
    evaluator = TinyFaceTest(meta)
    results = evaluator.test_identification(all_features, image_paths, ranks)
    results = {k: v for k, v in zip(['rank-{}'.format(r) for r in ranks], results)}
    results = {k: v * 100 for k, v in results.items()}
    print(f"TinyFace stage metric_total: {time.perf_counter() - stage_start:.3f}s")
    return results


class TinyFaceTest:
    def __init__(self, meta):
        self.meta = meta

    def get_key(self, image_path):
        return os.path.splitext(os.path.basename(image_path))[0]

    def get_label(self, image_path):
        return int(os.path.basename(image_path).split('_')[0])

    def init_proto(self, image_paths, probe_paths, match_paths, distractor_paths):
        index_dict = {}
        for i, image_path in enumerate(image_paths):
            index_dict[self.get_key(image_path)] = i

        self.indices_probe = np.array([index_dict[self.get_key(img)] for img in probe_paths])
        self.indices_match = np.array([index_dict[self.get_key(img)] for img in match_paths])
        self.indices_distractor = np.array([index_dict[self.get_key(img)] for img in distractor_paths])

        self.labels_probe = np.array([self.get_label(img) for img in probe_paths])
        self.labels_match = np.array([self.get_label(img) for img in match_paths])
        self.labels_distractor = np.array([-100 for img in distractor_paths])

        self.indices_gallery = np.concatenate([self.indices_match, self.indices_distractor])
        self.labels_gallery = np.concatenate([self.labels_match, self.labels_distractor])

    def test_identification(self, features, image_paths, ranks=[1, 5, 20]):
        assert len(image_paths) == len(features)
        assert len(image_paths) == len(self.meta['image_paths'])
        init_start = time.perf_counter()
        self.init_proto(image_paths,
                        self.meta['probe_paths'],
                        self.meta['gallery_paths'],
                        self.meta['distractor_paths'])
        print(f"TinyFace stage init_proto: {time.perf_counter() - init_start:.3f}s")

        stage_start = time.perf_counter()
        feat_probe = features[self.indices_probe]
        feat_gallery = features[self.indices_gallery]
        print(f"TinyFace stage index_features: {time.perf_counter() - stage_start:.3f}s")

        score_mat = inner_product_torch(feat_probe, feat_gallery)
        eval_method = os.environ.get('TINYFACE_EVAL_METHOD', 'closed_set')
        print(f"TinyFace evaluation method: {eval_method}")
        metric_start = time.perf_counter()
        if eval_method == 'closed_set':
            results = closed_set_cmc(
                score_mat=score_mat,
                probe_labels=self.labels_probe,
                gallery_labels=self.labels_gallery,
                ranks=ranks,
            )
            print(f"TinyFace stage closed_set_cmc: {time.perf_counter() - metric_start:.3f}s")
        elif eval_method == 'legacy':
            stage_start = time.perf_counter()
            label_mat = self.labels_probe[:, None] == self.labels_gallery[None, :]
            print(f"TinyFace stage label_matrix: {time.perf_counter() - stage_start:.3f}s")
            results, _, _ = DIR_FAR(score_mat, label_mat, ranks)
            print(f"TinyFace stage dir_far: {time.perf_counter() - metric_start:.3f}s")
        else:
            raise ValueError(
                f"Unsupported TINYFACE_EVAL_METHOD={eval_method!r}; "
                "expected 'closed_set' or 'legacy'"
            )

        return results


def inner_product_torch(x1, x2):
    """Use torch CPU ops to avoid numpy OpenMP thread pool deadlock."""
    num_threads = get_tinyface_num_threads()
    previous_threads = torch.get_num_threads()
    if previous_threads != num_threads:
        torch.set_num_threads(num_threads)
    try:
        print(
            f"TinyFace similarity: probes={len(x1)}, gallery={len(x2)}, "
            f"dim={x1.shape[1]}, cpu_threads={num_threads}"
        )
        stage_start = time.perf_counter()
        t1 = torch.from_numpy(x1).float()
        t2 = torch.from_numpy(x2).float()
        print(f"TinyFace stage tensor_conversion: {time.perf_counter() - stage_start:.3f}s")

        stage_start = time.perf_counter()
        t1 = t1 / t1.norm(dim=1, keepdim=True)
        t2 = t2 / t2.norm(dim=1, keepdim=True)
        print(f"TinyFace stage normalize: {time.perf_counter() - stage_start:.3f}s")

        stage_start = time.perf_counter()
        score = torch.mm(t1, t2.T)
        print(f"TinyFace stage matmul: {time.perf_counter() - stage_start:.3f}s")

        stage_start = time.perf_counter()
        score_numpy = score.numpy()
        print(f"TinyFace stage score_to_numpy: {time.perf_counter() - stage_start:.3f}s")
        return score_numpy
    finally:
        if previous_threads != num_threads:
            torch.set_num_threads(previous_threads)
