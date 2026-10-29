# -*- coding: utf-8 -*-
"""Independent dl_framework training entry for QttentionQCNN.
https://github.com/asdvfghg/QCNN_for_bearing_diagnosis
https://ieeexplore.ieee.org/abstract/document/10076833.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import FDPreprocess, FDTrainDataManager
from dl_framework import (
    AnalysisPolicy, ArtifactPolicy, AsyncPolicy, BatchPack, CachePolicy, CheckpointPolicy,
    ForwardInputs, ForwardResult, LoaderGroupSpec, LoaderSpec, LogPolicy, LossResult,
    ModelUnit, PathPolicy, PhaseSpec, RunConfig, StepContext, TrainerBuildResult, TrainerBuilder,
)

try:
    from .model import QttentionQCNN
except ImportError:
    from model import QttentionQCNN


CONFIG_PATH = str(PROJECT_ROOT / "configs" / "bearing_datasets_config_strict.yaml")


@dataclass
class DataConfig:
    window_size: int = 2048
    overlap: float = 0.5
    batch_size: int = 64
    num_workers: int = 0
    build_backend: str = "thread"
    build_workers: int = 4
    target_fs: Optional[int] = 25600
    signal_keys: Sequence[str] = ("x",)
    x_key: str = "x"
    y_key: str = "y"
    split_ratios: Sequence[float] = (0.7, 0.1, 0.2)
    split_names: Sequence[str] = ("train", "val", "test")
    normalize_method: str = "zoom"
    loader_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    epochs: int = 300
    lr: float = 1e-4
    weight_decay: float = 1e-4
    emb_dim: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "F:/Python/a01-papers/Paper1.1-jn/"
    task_prefix: str = "paper_reproductions"
    model_name: str = "official_qcnn"
    curve_every_n_epochs: int = 1000
    excel_every_n_epochs: int = 1000
    checkpoint_every_n_epochs: int = 1
    checkpoint_async: bool = False
    terminal_metric_include: Sequence[str] = ("*.loss", "*.classification.acc", "*.classification.f1_macro", "*.classification.mcc")
    terminal_metric_exclude: Sequence[str] = ("*.classification.class_*", "*.classification.label_*")
    excel_metric_include: Sequence[str] = ()
    excel_metric_exclude: Sequence[str] = ()
    data: DataConfig = field(default_factory=DataConfig)


def make_loader(ds: Any, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers)


def _metric_patterns(value: Sequence[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(x) for x in value if str(x).strip())


@dataclass
class ForwardAdapter:
    x_key: str = "x"
    y_key: str = "y"

    def build_inputs(self, unit: ModelUnit, batch_pack: BatchPack, ctx: StepContext) -> ForwardInputs:
        xs, ys, domains = [], [], []
        for view in batch_pack.batches.values():
            batch = view.data
            xs.append(batch[self.x_key])
            ys.append(batch[self.y_key])
            domains.extend([view.domain_name] * int(batch[self.y_key].shape[0]))
        return ForwardInputs(
            args=(torch.cat(xs, dim=0).to(unit.device),),
            kwargs={"return_feature_map": getattr(ctx, "phase_type", "") != "train"},
            meta={"target": torch.cat(ys, dim=0).long().to(unit.device), "domains": domains},
        )

    def forward(self, unit: ModelUnit, inputs: ForwardInputs, ctx: StepContext) -> ForwardResult:
        outputs = unit.model(*inputs.args, **inputs.kwargs)
        target = inputs.meta["target"]
        keep = getattr(ctx, "phase_type", "") != "train"
        return ForwardResult(
            main=outputs["logits"],
            loss_inputs={"outputs": outputs, "logits": outputs["logits"], "target": target},
            metric_inputs={"logits": outputs["logits"], "target": target},
            features={"features": outputs.get("features"), "feature_map": outputs.get("feature_map")} if keep else None,
        )


def loss_fn(unit_results: Mapping[str, ForwardResult], batch_pack: BatchPack, ctx: StepContext) -> LossResult:
    del batch_pack, ctx
    result = unit_results["main"]
    loss = F.cross_entropy(result.loss_inputs["logits"], result.loss_inputs["target"])
    return LossResult(total_loss=loss, scalars={"loss": float(loss.detach().cpu())})


def metric_fn(unit_results: Mapping[str, ForwardResult], batch_pack: BatchPack, ctx: StepContext) -> dict[str, Any]:
    del batch_pack, ctx
    result = unit_results["main"]
    return {"classification": {"y_true": result.metric_inputs["target"], "y_score": result.metric_inputs["logits"], "task": "single_label"}}


class QttentionQCNNTrainer:
    def __init__(self, config_path: str = CONFIG_PATH, cfg: Optional[TrainConfig] = None):
        self.config_path = config_path
        self.cfg = cfg or TrainConfig()
        self.fd_manager = FDTrainDataManager(config=config_path)
        self.build: Optional[TrainerBuildResult] = None

    def build_dataset(self, dataset_name: str, subset_name: Sequence[str] | str) -> Dict[str, Any]:
        keys = list(self.cfg.data.signal_keys)
        file_preprocess = []
        if self.cfg.data.target_fs is not None:
            file_preprocess.append(FDPreprocess.file_resample(target_fs=self.cfg.data.target_fs, source_fs_key="sampling_rate", keys=keys, axis=0))
        return self.fd_manager.build_and_split_and_to_torch_dataset(
            dataset_name=dataset_name,
            subset_name=[subset_name] if isinstance(subset_name, str) else subset_name,
            loader_kwargs=dict(self.cfg.data.loader_kwargs),
            file_preprocess=file_preprocess,
            post_parse_pipeline=[
                {"level": "dataset", "fns": [FDPreprocess.dataset_sliding_window(window_size=self.cfg.data.window_size, overlap=self.cfg.data.overlap, axis=0, keys=keys, drop_last=True)]},
                {"level": "sample", "fns": [FDPreprocess.sample_normalize(keys=keys, method=self.cfg.data.normalize_method, axis=-1), FDPreprocess.sample_expand_dims(axis=0, keys=keys)]},
            ],
            backend=self.cfg.data.build_backend,
            num_workers=self.cfg.data.build_workers,
            split_ratios=self.cfg.data.split_ratios,
            split_names=self.cfg.data.split_names,
            split_mode="random",
            split_group_by="label",
            shuffle_within_split=False,
            x_key=self.cfg.data.x_key,
            y_key=self.cfg.data.y_key,
            return_dict=True,
        )

    def _loader_factory(self, ds: Any, shuffle: bool):
        return lambda: make_loader(ds, self.cfg.data.batch_size, shuffle, self.cfg.data.num_workers)

    def _phase(self, name: str, phase_type: str, loader_name: str, factory: Any, role: str) -> PhaseSpec:
        loader = LoaderSpec(name=loader_name, loader_factory=factory, domain_name=loader_name, domain_role=role)
        return PhaseSpec(
            name=name,
            phase_type=phase_type,  # type: ignore[arg-type]
            loader_group=LoaderGroupSpec(mode="single", loaders=(loader,)),
            enable_backward=phase_type == "train",
            enable_optimizer_step=phase_type == "train",
            grad_clip_norm=1.0 if phase_type == "train" else None,
            allow_target_domain=phase_type != "train",
        )

    def build_trainer(self, dataset_name: str = "PU", train_subset: str = "ds1", test_subsets: Sequence[str] = ("ds1", "ds2", "ds3", "ds4")) -> TrainerBuildResult:
        fd_ds = self.build_dataset(dataset_name, train_subset)
        train_ds, val_ds = fd_ds["train"], fd_ds["val"]
        model = QttentionQCNN(
            win=int(getattr(train_ds, "window", self.cfg.data.window_size) or self.cfg.data.window_size),
            num_classes=int(train_ds.num_classes),
            in_channels=len(self.cfg.data.signal_keys),
            emb_dim=self.cfg.emb_dim,
            name=self.cfg.model_name,
        )
        phases = [
            self._phase("train", "train", f"{dataset_name}_{train_subset}_train", self._loader_factory(train_ds, True), "source"),
            self._phase("val", "val", f"{dataset_name}_{train_subset}_val", self._loader_factory(val_ds, False), "source"),
        ]
        for subset in test_subsets:
            ds = self.build_dataset(dataset_name, subset)
            loader = make_loader(ds["test"], self.cfg.data.batch_size, False, self.cfg.data.num_workers)
            lname = f"{dataset_name}_{subset}"
            phases.append(self._phase(f"test_{lname}", "test", lname, lambda loader=loader: loader, "target" if subset != train_subset else "source"))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        root_dir = Path(self.cfg.save_dir) / self.cfg.task_prefix / model.name
        config = RunConfig(
            experiment_name=f"{model.name}_{dataset_name}_{train_subset}",
            run_tag="train",
            max_epochs=self.cfg.epochs,
            device=self.cfg.device,
            phases=tuple(phases),
            path=PathPolicy(root_dir=str(root_dir)),
            async_policy=AsyncPolicy(backend="thread" if self.cfg.checkpoint_async else "sync", max_workers=2, queue_size=8, backpressure="block"),
            cache=CachePolicy(metric_prediction_max_samples=50_000),
            checkpoint=CheckpointPolicy(save_policy="best_and_last", monitor="val.classification.f1_macro", monitor_mode="max", save_every_n_epochs=max(1, self.cfg.checkpoint_every_n_epochs), async_save=self.cfg.checkpoint_async, keep_one_per_tag=True, save_optimizer=False),
            log=LogPolicy(
                save_csv=False,
                save_excel=True,
                excel_every_n_epochs=max(1, self.cfg.excel_every_n_epochs),
                excel_notes=(f"method={model.name}; dataset={dataset_name}; train_subset={train_subset}; test_subsets={list(test_subsets)}",),
                terminal_print_max_metrics=0,
                terminal_metric_include=_metric_patterns(self.cfg.terminal_metric_include),
                terminal_metric_exclude=_metric_patterns(self.cfg.terminal_metric_exclude),
                excel_metric_include=_metric_patterns(self.cfg.excel_metric_include),
                excel_metric_exclude=_metric_patterns(self.cfg.excel_metric_exclude),
                plot_on_end=True,
                plot_every_n_epochs=self.cfg.curve_every_n_epochs,
                curve_model_name=model.name,
                curve_dataset_name=f"{dataset_name}_{train_subset}",
            ),
            artifact=ArtifactPolicy(enabled=False),
            analysis=AnalysisPolicy(enabled=False, run_on_end=False),
        )
        unit = ModelUnit(name="main", model=model, optimizer=optimizer, forward_adapter=ForwardAdapter(x_key=self.cfg.data.x_key, y_key=self.cfg.data.y_key))
        self.build = TrainerBuilder(config).build(units={"main": unit}, loss_fn=loss_fn, metric_fns={"classification": metric_fn})
        return self.build

    def train(self, dataset_name: str = "JNU", train_subset: str = "ds1", test_subsets: Sequence[str] = ("ds1", "ds2", "ds3")) -> TrainerBuildResult:   #, "ds4"
        build = self.build_trainer(dataset_name, train_subset, test_subsets)
        build.engine.fit()
        print(f"Run key: {build.paths.run_id}")
        print(f"Metrics XLSX: {build.paths.excel_file('metrics')}")
        print(f"Models: {build.paths.model_dir}")
        return build


def train_qttention_qcnn(cfg: Optional[TrainConfig] = None) -> TrainerBuildResult:
    return QttentionQCNNTrainer(cfg=cfg).train()


if __name__ == "__main__":
    train_qttention_qcnn()
