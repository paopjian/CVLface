import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.distributed as distributed


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, '..', '..', '..', '..'))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, PACKAGE_ROOT)

from classifiers.partial_fc.partial_fc import PartialFC_V2
from classifiers.partial_fc import PartialFCClassifier
from losses.adaface import AdaFaceLoss, ContraFaceLoss
from losses.margin_loss import CombinedMarginLoss
from models.iresnet import IResNetModel
from pipelines.train_model_cls_pipeline import TrainModelClsPipeline


class SubcenterPartialFCTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dist_dir = tempfile.TemporaryDirectory()
        init_file = os.path.join(cls.dist_dir.name, 'process_group')
        os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo')
        distributed.init_process_group(
            backend='gloo',
            init_method=f'file://{init_file}',
            rank=0,
            world_size=1,
        )

    @classmethod
    def tearDownClass(cls):
        distributed.destroy_process_group()
        cls.dist_dir.cleanup()

    @staticmethod
    def make_classifier(margin_loss, num_subcenters=3, sample_rate=1.0):
        return PartialFC_V2(
            rank=0,
            world_size=1,
            margin_loss=margin_loss,
            embedding_size=2,
            num_classes=3,
            sample_rate=sample_rate,
            num_subcenters=num_subcenters,
        )

    def test_subcenter_logits_take_per_class_maximum(self):
        classifier = self.make_classifier(
            CombinedMarginLoss(64, 1.0, 0.5, 0.0),
            num_subcenters=3,
        )
        embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        weights = torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0],
            [0.6, 0.8], [-1.0, 0.0], [0.0, -1.0],
        ])

        logits = classifier.compute_logits(embeddings, weights)

        expected = torch.tensor([[1.0, 0.6], [1.0, 0.8]])
        torch.testing.assert_close(logits, expected)

    def test_sampling_keeps_all_subcenters_for_each_identity(self):
        classifier = self.make_classifier(
            CombinedMarginLoss(64, 1.0, 0.5, 0.0),
            num_subcenters=3,
            sample_rate=0.34,
        )
        labels = torch.tensor([[1], [-1]])
        index_positive = labels != -1

        sampled_weight = classifier.sample(labels, index_positive)

        self.assertEqual(classifier.weight_index.tolist(), [1])
        self.assertEqual(classifier.subcenter_weight_index.tolist(), [3, 4, 5])
        self.assertEqual(sampled_weight.shape, (3, 2))
        self.assertEqual(labels.tolist(), [[0], [-1]])

    def test_k1_matches_single_center_cosine_logits(self):
        classifier = self.make_classifier(
            CombinedMarginLoss(64, 1.0, 0.5, 0.0),
            num_subcenters=1,
        )
        embeddings = torch.randn(4, 2)
        weights = torch.randn(3, 2)

        actual = classifier.compute_logits(embeddings, weights)
        expected = torch.nn.functional.linear(
            torch.nn.functional.normalize(embeddings, dim=1),
            torch.nn.functional.normalize(weights, dim=1),
        )

        torch.testing.assert_close(actual, expected)

    def test_adaface_subcenter_forward_and_backward(self):
        classifier = self.make_classifier(
            AdaFaceLoss(s=64, m=0.4, h=0.333, t_alpha=0.01),
            num_subcenters=3,
        )
        embeddings = torch.tensor(
            [[1.0, 0.2], [0.1, 1.0], [-0.8, 0.3], [0.4, -0.9]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 1, 2, 0])

        loss = classifier(embeddings, labels)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        self.assertTrue(torch.isfinite(classifier.weight.grad).all())
        self.assertGreater(classifier.weight.grad.abs().sum().item(), 0)

    def test_coreface_two_views_with_subcenter_backward(self):
        classifier = self.make_classifier(
            AdaFaceLoss(s=64, m=0.4, h=0.333, t_alpha=0.01),
            num_subcenters=3,
        )
        feature1 = torch.randn(4, 2, requires_grad=True)
        feature2 = torch.randn(4, 2, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 0])

        loss1 = classifier(feature1, labels.clone())
        loss2 = classifier(feature2, labels.clone())
        contrast = ContraFaceLoss()(feature1, feature2, labels)
        loss = 0.5 * loss1 + 0.5 * loss2 + 0.05 * contrast
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(feature1.grad).all())
        self.assertTrue(torch.isfinite(feature2.grad).all())
        self.assertTrue(torch.isfinite(classifier.weight.grad).all())

    def test_coreface_subcenter_pipeline_backward(self):
        model_config = SimpleNamespace(
            name='ir18',
            output_dim=8,
            color_space='RGB',
            freeze=False,
            dropout=0.4,
        )
        classifier_config = SimpleNamespace(
            name='partial_fc',
            sample_rate=1.0,
            num_subcenters=3,
            freeze=False,
        )
        pipeline_config = SimpleNamespace(
            coreface_enabled=True,
            coreface_start_epoch=0,
            coreface_dropout=0.4,
            coreface_dropout2=0.4,
            coreface_weight1=0.5,
            coreface_weight2=0.5,
            coreface_weight_contrast=0.05,
            coreface_weight_contrast_reverse=0.0,
        )
        model = IResNetModel.from_config(model_config)
        classifier = PartialFCClassifier.from_config(
            classifier_config,
            AdaFaceLoss(s=64, m=0.4, h=0.333, t_alpha=0.01),
            model_config,
            num_classes=3,
            rank=0,
            world_size=1,
        )
        pipeline = TrainModelClsPipeline(
            model,
            classifier,
            optimizer=None,
            lr_scheduler=None,
            pipeline_config=pipeline_config,
        )
        pipeline.set_epoch(0)

        loss = pipeline((
            torch.randn(2, 3, 112, 112),
            torch.tensor([0, 1]),
        ))
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIn('train/coreface_loss_view1', pipeline.last_losses)
        self.assertIn('train/coreface_loss_contrast', pipeline.last_losses)
        self.assertTrue(torch.isfinite(classifier.partial_fc.weight.grad).all())

    def test_classifier_factory_reads_num_subcenters(self):
        config = SimpleNamespace(
            name='partial_fc',
            sample_rate=0.4,
            num_subcenters=3,
        )
        model_config = SimpleNamespace(output_dim=2)

        classifier = PartialFCClassifier.from_config(
            config,
            AdaFaceLoss(s=64, m=0.4, h=0.333, t_alpha=0.01),
            model_config,
            num_classes=5,
            rank=0,
            world_size=1,
        )

        self.assertEqual(classifier.partial_fc.num_subcenters, 3)
        self.assertEqual(classifier.partial_fc.weight.shape, (15, 2))

    def test_checkpoint_redistribution_preserves_class_boundaries(self):
        partial_fc = PartialFC_V2(
            rank=0,
            world_size=1,
            margin_loss=AdaFaceLoss(s=64, m=0.4, h=0.333, t_alpha=0.01),
            embedding_size=2,
            num_classes=5,
            num_subcenters=3,
        )
        classifier = PartialFCClassifier(partial_fc, {}, rank=0, world_size=1)
        combined_weight = torch.arange(36, dtype=torch.float32).reshape(18, 2)

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            for rank in range(2):
                state_dict = {
                    'partial_fc.weight': combined_weight[rank * 9:(rank + 1) * 9],
                    'partial_fc.batch_mean': torch.tensor([20.0]),
                    'partial_fc.batch_std': torch.tensor([100.0]),
                }
                torch.save(
                    state_dict,
                    os.path.join(checkpoint_dir, f'classifier_rank{rank}.pt'),
                )

            classifier.load_state_dict_from_path(checkpoint_dir)

        torch.testing.assert_close(classifier.partial_fc.weight, combined_weight[:15])


if __name__ == '__main__':
    unittest.main()
