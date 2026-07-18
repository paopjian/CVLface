import torch

from losses.qgface import QGFaceLoss
from models.base import BaseModel

from .base import BasePipeline


class TrainQGFacePipeline(BasePipeline):
    def __init__(
        self,
        model: BaseModel,
        classifier,
        model_optimizer,
        classifier_optimizer,
        model_lr_scheduler,
        classifier_lr_scheduler,
        fabric,
        config,
    ):
        super().__init__()
        if classifier is None:
            raise ValueError("QGFace requires a classification head")

        self.model = model
        self.classifier = classifier
        self.model_optimizer = model_optimizer
        self.classifier_optimizer = classifier_optimizer
        self.model_lr_scheduler = model_lr_scheduler
        self.classifier_lr_scheduler = classifier_lr_scheduler
        self.fabric = fabric
        self.base_contrast_weight = config.contrast_weight
        self.contrast_start_epoch = config.contrast_start_epoch
        self.contrast_weight = (
            0.0 if self.contrast_start_epoch > 0 else self.base_contrast_weight
        )
        self.qgface_loss = QGFaceLoss(
            embedding_size=model.config.output_dim,
            **dict(config.qgface),
        ).to(model.device)
        self.last_metrics = {}

    @property
    def module_names_list(self):
        return [
            "model",
            "classifier",
            "qgface_loss",
            "model_optimizer",
            "classifier_optimizer",
            "model_lr_scheduler",
            "classifier_lr_scheduler",
        ]

    def integrity_check(self, dataset):
        if not getattr(dataset, "contrastive_views", False):
            raise ValueError("QGFace requires data_augs.contrastive_views=true")
        if dataset.color_space != self.model.config.color_space:
            raise ValueError(
                f"Dataset/model color space mismatch: {dataset.color_space}/{self.model.config.color_space}"
            )
        self.color_space = dataset.color_space
        self.make_train_transform()

    def make_train_transform(self):
        return self.model.make_train_transform()

    def set_epoch(self, epoch):
        self.contrast_weight = (
            self.base_contrast_weight if epoch >= self.contrast_start_epoch else 0.0
        )

    @torch.no_grad()
    def _gather_tensor(self, tensor):
        if self.fabric.world_size == 1:
            return tensor.detach()
        gathered = self.fabric.all_gather(tensor.detach())
        return gathered.reshape(-1, *tensor.shape[1:])

    @torch.no_grad()
    def _classifier_method(self, name, *args):
        module = self.classifier
        for _ in range(4):
            method = getattr(module, name, None)
            if callable(method):
                return method(*args)
            module = getattr(module, "module", getattr(module, "_forward_module", None))
            if module is None:
                break
        raise TypeError(f"Classifier does not implement {name}()")

    def __call__(self, batch):
        if len(batch) != 3:
            raise ValueError(
                "QGFace batches must contain augmented image, original image and label"
            )
        augmented_images, original_images, labels = batch
        labels = labels.to(self.model.device).long()
        batch_size = labels.shape[0]

        images = torch.cat([original_images, augmented_images], dim=0)
        embeddings = self.model(images)
        original_embeddings, augmented_embeddings = embeddings.split(batch_size)
        original_norms = original_embeddings.norm(
            p=2, dim=1, keepdim=True
        ).clamp_min(1e-8)
        augmented_norms = augmented_embeddings.norm(
            p=2, dim=1, keepdim=True
        ).clamp_min(1e-8)
        classification_labels = torch.cat([labels, labels], dim=0)
        classification_loss = self.classifier(embeddings, classification_labels)

        if self.contrast_weight > 0:
            augmented_scalers = self._classifier_method(
                "get_margin_scaler", augmented_norms
            )
            original_scalers = self._classifier_method(
                "get_margin_scaler", original_norms
            )

            with torch.no_grad():
                all_original = self._gather_tensor(original_embeddings)
                all_augmented = self._gather_tensor(augmented_embeddings)
                all_labels = self._gather_tensor(labels)
                queue_batch_embeddings = torch.cat(
                    [all_original, all_augmented], dim=0
                )
                queue_batch_labels = torch.cat([all_labels, all_labels], dim=0)
                if queue_batch_embeddings.shape[0] > self.qgface_loss.queue_size:
                    raise ValueError(
                        "QGFace queue_size must be at least twice the global batch size"
                    )

                batch_proxies = self._classifier_method(
                    "get_class_proxies", queue_batch_labels
                )
                written_indices = self.qgface_loss.enqueue(
                    queue_batch_embeddings,
                    queue_batch_labels,
                    batch_proxies,
                )
                _, queued_labels = self.qgface_loss.get_queue()
                current_proxies = self._classifier_method(
                    "get_class_proxies", queued_labels
                )
                queue_embeddings, queued_labels = self.qgface_loss.get_queue(
                    current_proxies
                )

                global_batch_size = all_labels.shape[0]
                local_offset = self.fabric.global_rank * batch_size
                local_indices = (
                    torch.arange(batch_size, device=labels.device) + local_offset
                )
                positive_indices = torch.stack(
                    [
                        written_indices[local_indices],
                        written_indices[global_batch_size + local_indices],
                    ],
                    dim=1,
                )

            contrastive_loss = self.qgface_loss(
                augmented_embeddings,
                original_embeddings,
                augmented_norms,
                original_norms,
                labels,
                queue_embeddings,
                queued_labels,
                positive_indices,
                margin_scalers=(augmented_scalers, original_scalers),
            )
        else:
            contrastive_loss = classification_loss.new_zeros(())
            self.qgface_loss.last_metrics = {
                "qgface/contrastive_loss": contrastive_loss,
                "qgface/positive_similarity": contrastive_loss,
                "qgface/quality_weight": contrastive_loss,
                "qgface/selected_norm": contrastive_loss,
                "qgface/queue_size": contrastive_loss,
            }

        loss = classification_loss + self.contrast_weight * contrastive_loss
        self.last_metrics = {
            "qgface/classification_loss": classification_loss.detach(),
            "qgface/contrast_weight": torch.tensor(
                self.contrast_weight, device=loss.device, dtype=torch.float32
            ),
            **self.qgface_loss.last_metrics,
        }
        return loss

    def get_log_dict(self):
        return self.last_metrics

    def train(self):
        if not self.model.config.freeze:
            self.model.train()
        else:
            self.model.eval()
        if not self.classifier.config.freeze:
            self.classifier.train()
        self.qgface_loss.train()

    def eval(self):
        self.model.eval()
        self.classifier.eval()
        self.qgface_loss.eval()
