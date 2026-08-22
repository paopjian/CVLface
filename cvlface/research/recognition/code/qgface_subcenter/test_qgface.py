from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import pyrootutils
import torch
from lightning import Fabric

pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)

import config
from classifiers import get_classifier
from dataset.contrastive_view_dataset import ContrastiveViewDataset
from dataset.qgface_view_transform import QGFaceViewTransform
from losses import get_margin_loss
from losses.qgface import QGFaceLoss
from models import get_model
from optims.optims import make_qgface_optimizers
from pipelines.train_qgface_pipeline import TrainQGFacePipeline


class TinyDataset(torch.utils.data.Dataset):
    color_space = "RGB"

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return torch.full((3, 4, 4), float(index)), index


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3 * 4 * 4, 8)
        self.config = SimpleNamespace(output_dim=8, color_space="RGB", freeze=False)

    @property
    def device(self):
        return self.linear.weight.device

    def forward(self, images):
        return self.linear(images.flatten(1))

    def make_train_transform(self):
        return None

    def has_trainable_params(self):
        return any(parameter.requires_grad for parameter in self.parameters())


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 3, 8))
        self.config = SimpleNamespace(freeze=False)
        self.last_batch_size = 0

    def forward(self, embeddings, labels):
        self.last_batch_size = embeddings.shape[0]
        features = torch.nn.functional.normalize(embeddings, dim=1)
        centers = torch.nn.functional.normalize(self.weight, dim=2)
        logits = torch.einsum("bd,ckd->bck", features, centers).amax(dim=2)
        return torch.nn.functional.cross_entropy(logits, labels)

    @torch.no_grad()
    def route_subcenters(self, labels, embeddings):
        features = torch.nn.functional.normalize(embeddings, dim=1)
        centers = torch.nn.functional.normalize(self.weight[labels], dim=2)
        return torch.einsum("bd,bkd->bk", features, centers).argmax(dim=1)

    @torch.no_grad()
    def get_class_proxies(self, labels, subcenter_ids):
        return self.weight[labels, subcenter_ids]

    @staticmethod
    def get_margin_scaler(norms):
        return torch.zeros_like(norms)

    def has_trainable_params(self):
        return any(parameter.requires_grad for parameter in self.parameters())


class TinyFabric:
    world_size = 1
    global_rank = 0
    local_rank = 0


def test_loss_and_queue():
    loss_fn = QGFaceLoss(
        embedding_size=8,
        queue_size=5,
        quality_scale=0,
        pair_coupling="D2N",
    )
    query = torch.randn(3, 8, requires_grad=True)
    key = torch.randn(3, 8)
    norms = query.detach().norm(dim=1, keepdim=True)
    key_norms = key.norm(dim=1, keepdim=True)
    labels = torch.arange(3)
    positive_indices = torch.arange(3)

    loss = loss_fn(
        query,
        key,
        norms,
        key_norms,
        labels,
        key,
        labels,
        positive_indices,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert query.grad is not None and torch.isfinite(query.grad).all()

    proxies = torch.randn(3, 8)
    subcenter_ids = torch.tensor([0, 1, 2])
    loss_fn.enqueue(key, labels, subcenter_ids, proxies)
    loss_fn.enqueue(key + 1, labels, subcenter_ids, proxies)
    queued_embeddings, queued_labels, queued_subcenter_ids = loss_fn.get_queue()
    assert queued_embeddings.shape == (5, 8)
    assert queued_labels.shape == (5,)
    assert torch.equal(queued_subcenter_ids, torch.tensor([2, 1, 2, 0, 1]))

    current_proxies = loss_fn.queue_proxies[:5] + 0.1
    updated_embeddings, _, _ = loss_fn.get_queue(current_proxies)
    assert not torch.allclose(updated_embeddings, queued_embeddings)


def test_split_optimizers():
    model = TinyModel()
    classifier = TinyClassifier()
    cfg = SimpleNamespace(
        optims=SimpleNamespace(
            optimizer="sgd",
            lr=0.2,
            classifier_lr=0.05,
            momentum=0.9,
            weight_decay=0.0005,
            classifier_weight_decay=0.0001,
            filter_bias_and_bn=True,
        )
    )
    model_optimizer, classifier_optimizer = make_qgface_optimizers(
        cfg, model, classifier
    )

    model_param_ids = {
        id(parameter)
        for group in model_optimizer.param_groups
        for parameter in group["params"]
    }
    classifier_param_ids = {
        id(parameter)
        for group in classifier_optimizer.param_groups
        for parameter in group["params"]
    }
    assert model_param_ids
    assert classifier_param_ids
    assert model_param_ids.isdisjoint(classifier_param_ids)
    assert model_optimizer.param_groups[0]["lr"] == 0.2
    assert classifier_optimizer.param_groups[0]["lr"] == 0.05
    assert classifier_optimizer.param_groups[0]["weight_decay"] == 0.0001


def test_pipeline():
    dataset = ContrastiveViewDataset(
        TinyDataset(),
        view_transform=QGFaceViewTransform(output_size=4, jpeg_quality=75),
    )
    source_image, _ = dataset.dataset[1]
    _, original_image, _ = dataset[1]
    assert torch.equal(original_image, source_image) or torch.equal(
        original_image, source_image.flip(-1)
    )
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))
    model = TinyModel()
    classifier = TinyClassifier()
    model_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    classifier_optimizer = torch.optim.SGD(classifier.parameters(), lr=0.05)
    config = SimpleNamespace(
        contrast_weight=1.0,
        contrast_start_epoch=0,
        qgface={
            "queue_size": 8,
            "quality_scale": 0,
            "pair_coupling": "D2N",
        },
    )
    pipeline = TrainQGFacePipeline(
        model,
        classifier,
        model_optimizer,
        classifier_optimizer,
        None,
        None,
        TinyFabric(),
        config,
    )
    pipeline.integrity_check(dataset)
    loss = pipeline(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert int(pipeline.qgface_loss.queue_valid_size.item()) == 8
    assert torch.equal(
        pipeline.qgface_loss.queue_subcenter_ids[:4],
        pipeline.qgface_loss.queue_subcenter_ids[4:8],
    )
    assert classifier.last_batch_size == 8
    assert model.linear.weight.grad is not None
    assert "qgface/contrastive_loss" in pipeline.get_log_dict()


def test_classifier_only_pipeline():
    dataset = ContrastiveViewDataset(TinyDataset())
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))
    model = TinyModel()
    classifier = TinyClassifier()
    model_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    classifier_optimizer = torch.optim.SGD(classifier.parameters(), lr=0.05)
    config = SimpleNamespace(
        contrast_weight=0.0,
        contrast_start_epoch=0,
        qgface={"queue_size": 8, "quality_scale": 0},
    )
    pipeline = TrainQGFacePipeline(
        model,
        classifier,
        model_optimizer,
        classifier_optimizer,
        None,
        None,
        TinyFabric(),
        config,
    )
    pipeline.integrity_check(dataset)
    loss = pipeline(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert int(pipeline.qgface_loss.queue_valid_size.item()) == 0
    assert pipeline.get_log_dict()["qgface/contrastive_loss"].item() == 0


def test_split_optimizer_checkpoint():
    model = TinyModel()
    classifier = TinyClassifier()
    model_optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    classifier_optimizer = torch.optim.SGD(
        classifier.parameters(), lr=0.05, momentum=0.9
    )
    config = SimpleNamespace(
        contrast_weight=0.0,
        contrast_start_epoch=0,
        qgface={"queue_size": 8, "quality_scale": 0},
    )
    pipeline = TrainQGFacePipeline(
        model,
        classifier,
        model_optimizer,
        classifier_optimizer,
        None,
        None,
        TinyFabric(),
        config,
    )
    pipeline.color_space = "RGB"

    with TemporaryDirectory() as temp_dir:
        pipeline.save_pipelines_and_configs(
            temp_dir,
            TinyFabric(),
            pipeline,
            {"test": True},
            epoch=2,
            step=10,
            n_images_seen=40,
        )
        saved_files = {path.name for path in Path(temp_dir).iterdir()}
        assert "model_optimizer.pt" in saved_files
        assert "classifier_optimizer.pt" in saved_files
        assert "optimizer.pt" not in saved_files

        resumed_model = TinyModel()
        resumed_classifier = TinyClassifier()
        resumed_pipeline = TrainQGFacePipeline(
            resumed_model,
            resumed_classifier,
            torch.optim.SGD(resumed_model.parameters(), lr=0.1, momentum=0.9),
            torch.optim.SGD(
                resumed_classifier.parameters(), lr=0.05, momentum=0.9
            ),
            None,
            None,
            TinyFabric(),
            config,
        )
        epoch, step, n_images_seen = resumed_pipeline.resume_from_dir(temp_dir)
        assert (epoch, step, n_images_seen) == (2, 10, 40)


def test_subcenter_partial_fc():
    with TemporaryDirectory() as temp_dir:
        init_file = Path(temp_dir) / "distributed_init"
        torch.distributed.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=0,
            world_size=1,
        )
        try:
            classifier_config = config.load_yaml(
                "pfc40_subcenter_k3", directory="classifiers"
            )
            classifier_config.sample_rate = 1.0
            loss_config = config.load_yaml("qgface_adaface", directory="losses")
            classifier = get_classifier(
                classifier_config,
                get_margin_loss(loss_config),
                SimpleNamespace(output_dim=8),
                num_classes=4,
                rank=0,
                world_size=1,
            )

            embeddings = torch.randn(2, 8, requires_grad=True)
            labels = torch.tensor([0, 1])
            loss = classifier(embeddings, labels)
            loss.backward()
            assert torch.isfinite(loss)
            assert classifier.partial_fc.weight.shape == (12, 8)
            assert classifier.partial_fc.weight.grad is not None

            subcenter_ids = classifier.route_subcenters(labels, embeddings.detach())
            proxies = classifier.get_class_proxies(labels, subcenter_ids)
            expected_indices = labels * 3 + subcenter_ids
            assert torch.equal(proxies, classifier.partial_fc.weight[expected_indices])
        finally:
            torch.distributed.destroy_process_group()


def test_real_components_with_fabric():
    model_config = config.load_yaml("iresnet.v1_ir18", directory="models")
    classifier_config = config.load_yaml("fc", directory="classifiers")
    loss_config = config.load_yaml("qgface_adaface", directory="losses")
    pipeline_config = config.load_yaml("train_qgface", directory="pipelines")
    pipeline_config.qgface.queue_size = 8
    pipeline_config.qgface.quality_scale = 0

    model = get_model(model_config, task="qgface_test")
    classifier = get_classifier(
        classifier_config,
        get_margin_loss(loss_config),
        model_config,
        num_classes=4,
        rank=0,
        world_size=1,
    )

    model_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    classifier_optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01)
    fabric = Fabric(accelerator="cpu", devices=1, strategy="auto", precision="32-true")
    fabric.launch()
    model, model_optimizer = fabric.setup(model, model_optimizer)
    classifier = fabric.setup(classifier)
    classifier_optimizer = fabric.setup_optimizers(classifier_optimizer)
    pipeline = TrainQGFacePipeline(
        model,
        classifier,
        model_optimizer,
        classifier_optimizer,
        None,
        None,
        fabric,
        pipeline_config,
    )

    images = torch.randn(2, 3, 112, 112, device=fabric.device)
    labels = torch.tensor([0, 1], device=fabric.device)
    model_parameter = next(model.parameters())
    classifier_parameter = next(classifier.parameters())
    model_before = model_parameter.detach().clone()
    classifier_before = classifier_parameter.detach().clone()
    loss = pipeline((images, images.flip(-1), labels))
    fabric.backward(loss)
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())
    fabric.clip_gradients(model, model_optimizer, max_norm=5.0)
    fabric.clip_gradients(classifier, classifier_optimizer, max_norm=5.0)
    model_optimizer.step()
    classifier_optimizer.step()
    assert not torch.equal(model_parameter.detach(), model_before)
    assert not torch.equal(classifier_parameter.detach(), classifier_before)


if __name__ == "__main__":
    test_loss_and_queue()
    test_split_optimizers()
    test_pipeline()
    test_classifier_only_pipeline()
    test_split_optimizer_checkpoint()
    test_subcenter_partial_fc()
    test_real_components_with_fabric()
    print("QGFace tests passed")
