# -*- coding: utf-8 -*-
"""dl_framework training entry for the MWKG-Net paper reconstruction."""
from __future__ import annotations

import math
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import FDPreprocess, FDTrainDataManager, Operation
from dl_framework import (
    AnalysisPolicy,
    ArtifactPolicy,
    AsyncPolicy,
    BatchPack,
    CachePolicy,
    CheckpointPolicy,
    ForwardInputs,
    ForwardResult,
    LoaderGroupSpec,
    LoaderSpec,
    LogPolicy,
    LossResult,
    ModelUnit,
    PathPolicy,
    PhaseSpec,
    RunConfig,
    StepContext,
    TrainerBuildResult,
    TrainerBuilder,
)

try:
    from .model import MWKGLoss, MWKGNet
except ImportError:
    from model import MWKGLoss, MWKGNet


CONFIG_PATH = str(PROJECT_ROOT / "configs" / "bearing_datasets_config_strict.yaml")


@dataclass
class DataConfig:
    # JNU operating-condition protocol used by this training entry:
    # ds1=600 rpm is the source domain; ds2=800 rpm and ds3=1000 rpm are
    # evaluated in full as unseen operating-condition target domains.
    dataset_name: str = "PU"
    train_subset: str = "ds1"
    test_subsets: Sequence[str] = ("ds1", "ds2", "ds3", "ds4")
    window_size: int = 2048
    overlap: float = 0.5
    batch_size: int = 64
    num_workers: int = 0
    build_backend: str = "thread"
    build_workers: int = 4
    target_fs: Optional[int] = None
    signal_keys: Sequence[str] = ("x",)
    x_key: str = "x"
    y_key: str = "y"
    # Paper states 70/30 train/test but also selects the best validation model.
    # We preserve 70% training and split the remaining 30% equally for val/test.
    split_ratios: Sequence[float] = (0.7, 0.15, 0.15)
    split_names: Sequence[str] = ("train", "val", "test")
    normalize_method: str = "zoom"
    # Clean JNU is the safe default. For the paper's independent robustness
    # experiments set one of -10,-5,0,5,10 per run; noise is never implicit.
    snr_db: Optional[float] = None
    noise_seed: int = 3407
    loader_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 0.0
    min_lr_ratio: float = 0.01
    use_cosine_lr: bool = True
    base_channels: int = 64
    deep_channels: int = 128
    wavelet_levels: int = 4
    wavelet_kernel_size: int = 5
    shallow_blocks: int = 2
    deep_blocks: int = 3
    kgfr_temperature: float = 1.0
    # Eq. (14) includes gamma, but the paper does not publish its value.
    kurtosis_gamma: float = 0.5
    grad_clip_norm: Optional[float] = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "F:/Python/a01-papers/Paper1.1-pu/"
    task_prefix: str = "paper_reproductions"
    model_name: str = "mwkg_net"
    seed: int = 3437
    curve_every_n_epochs: Optional[int] = None
    excel_every_n_epochs: int = 200
    checkpoint_every_n_epochs: int = 200
    checkpoint_async: bool = False
    terminal_metric_include: Sequence[str] = (
        "*.loss",
        "*.loss_*",
        "*.kurtosis_weight",
        "*.classification.acc",
        "*.classification.f1_macro",
        "*.classification.mcc",
    )
    terminal_metric_exclude: Sequence[str] = (
        "*.classification.class_*",
        "*.classification.label_*",
    )
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
    return tuple(str(item) for item in value if str(item).strip())


def sample_awgn_operation(*, keys: Sequence[str], snr_db: float, seed: int) -> Operation:
    """Add deterministic AWGN at the requested signal-to-noise ratio."""

    def _op(sample: Mapping[str, Any]) -> Dict[str, Any]:
        output = dict(sample)
        output["x_dict"] = dict(sample["x_dict"])
        sample_id = str(sample.get("sample_id", ""))
        local_seed = (int(seed) + zlib.crc32(sample_id.encode("utf-8"))) % (2**32)
        rng = np.random.default_rng(local_seed)
        for key in keys:
            if key not in output["x_dict"]:
                continue
            x = np.asarray(output["x_dict"][key], dtype=np.float32)
            signal_power = float(np.mean(np.square(x)))
            noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
            if signal_power <= 0.0 or noise_power <= 0.0:
                noisy = x.copy()
            else:
                noise = rng.normal(0.0, math.sqrt(noise_power), size=x.shape).astype(np.float32)
                noisy = x + noise
            output["x_dict"][key] = noisy.astype(np.float32, copy=False)
        meta = dict(sample.get("meta", {}))
        meta["awgn_snr_db"] = float(snr_db)
        output["meta"] = meta
        return output

    return Operation(
        _op,
        params={"keys": list(keys), "snr_db": float(snr_db), "seed": int(seed)},
        display_name="sample_add_awgn",
    )


@dataclass
class ForwardAdapter:
    x_key: str = "x"
    y_key: str = "y"

    def build_inputs(self, unit: ModelUnit, batch_pack: BatchPack, ctx: StepContext) -> ForwardInputs:
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        for view in batch_pack.batches.values():
            batch = view.data
            xs.append(batch[self.x_key])
            ys.append(batch[self.y_key])
        return ForwardInputs(
            args=(torch.cat(xs, dim=0).to(unit.device),),
            kwargs={"return_feature_map": getattr(ctx, "phase_type", "") != "train"},
            meta={"target": torch.cat(ys, dim=0).long().to(unit.device)},
        )

    def forward(self, unit: ModelUnit, inputs: ForwardInputs, ctx: StepContext) -> ForwardResult:
        outputs = unit.model(*inputs.args, **inputs.kwargs)
        target = inputs.meta["target"]
        keep = getattr(ctx, "phase_type", "") != "train"
        return ForwardResult(
            main=outputs["logits"],
            loss_inputs={"outputs": outputs, "target": target},
            metric_inputs={"logits": outputs["logits"], "target": target},
            features={
                "features": outputs["features"],
                "feature_map": outputs.get("feature_map"),
            }
            if keep
            else None,
            attentions={"channel_weights": outputs["channel_weights"]} if keep else None,
        )


def metric_fn(
    unit_results: Mapping[str, ForwardResult], batch_pack: BatchPack, ctx: StepContext
) -> dict[str, Any]:
    del batch_pack, ctx
    result = unit_results["main"]
    return {
        "classification": {
            "y_true": result.metric_inputs["target"],
            "y_score": result.metric_inputs["logits"],
            "task": "single_label",
        }
    }


class MWKGNetTrainer:
    def __init__(self, config_path: str = CONFIG_PATH, cfg: Optional[TrainConfig] = None):
        self.config_path = config_path
        self.cfg = cfg or TrainConfig()
        if len(self.cfg.data.signal_keys) != 1:
            raise ValueError("The paper architecture expects one vibration channel")
        if self.cfg.data.window_size % (2 ** self.cfg.wavelet_levels) != 0:
            raise ValueError("window_size must be divisible by 2 ** wavelet_levels")
        if not self.cfg.data.dataset_name.strip():
            raise ValueError("dataset_name must not be empty")
        if not self.cfg.data.train_subset.strip():
            raise ValueError("train_subset must not be empty")
        if not self.cfg.data.test_subsets:
            raise ValueError("test_subsets must contain at least one subset")
        if len(set(self.cfg.data.test_subsets)) != len(self.cfg.data.test_subsets):
            raise ValueError("test_subsets must not contain duplicates")
        self.fd_manager = FDTrainDataManager(config=config_path)
        self.loss_module = MWKGLoss(
            total_epochs=self.cfg.epochs,
            gamma=self.cfg.kurtosis_gamma,
        )
        self.build: Optional[TrainerBuildResult] = None

    def build_dataset(
        self,
        dataset_name: str,
        subset_name: Sequence[str] | str,
        *,
        split_ratios: Optional[Sequence[float]] = None,
        split_names: Optional[Sequence[str]] = None,
        split_mode: str = "random",
    ) -> Dict[str, Any]:
        keys = list(self.cfg.data.signal_keys)
        file_preprocess = []
        if self.cfg.data.target_fs is not None:
            file_preprocess.append(
                FDPreprocess.file_resample(
                    target_fs=self.cfg.data.target_fs,
                    source_fs_key="sampling_rate",
                    keys=keys,
                    axis=0,
                )
            )
        sample_ops = []
        if self.cfg.data.snr_db is not None:
            sample_ops.append(
                sample_awgn_operation(
                    keys=keys,
                    snr_db=self.cfg.data.snr_db,
                    seed=self.cfg.data.noise_seed,
                )
            )
        sample_ops.extend(
            [
                FDPreprocess.sample_normalize(
                    keys=keys,
                    method=self.cfg.data.normalize_method,
                    axis=-1,
                ),
                FDPreprocess.sample_expand_dims(axis=0, keys=keys),
            ]
        )
        return self.fd_manager.build_and_split_and_to_torch_dataset(
            dataset_name=dataset_name,
            subset_name=[subset_name] if isinstance(subset_name, str) else subset_name,
            loader_kwargs=dict(self.cfg.data.loader_kwargs),
            file_preprocess=file_preprocess,
            post_parse_pipeline=[
                {
                    "level": "dataset",
                    "fns": [
                        FDPreprocess.dataset_sliding_window(
                            window_size=self.cfg.data.window_size,
                            overlap=self.cfg.data.overlap,
                            axis=0,
                            keys=keys,
                            drop_last=True,
                        )
                    ],
                },
                {"level": "sample", "fns": sample_ops},
            ],
            backend=self.cfg.data.build_backend,
            num_workers=self.cfg.data.build_workers,
            split_ratios=tuple(split_ratios or self.cfg.data.split_ratios),
            split_names=tuple(split_names or self.cfg.data.split_names),
            split_mode=split_mode,
            split_group_by="label",
            shuffle_within_split=False,
            seed=self.cfg.seed,
            x_key=self.cfg.data.x_key,
            y_key=self.cfg.data.y_key,
            return_dict=True,
        )

    def _loader_factory(self, ds: Any, shuffle: bool):
        return lambda: make_loader(ds, self.cfg.data.batch_size, shuffle, self.cfg.data.num_workers)

    def _phase(self, name: str, phase_type: str, loader_name: str, factory: Any, role: str) -> PhaseSpec:
        loader = LoaderSpec(
            name=loader_name,
            loader_factory=factory,
            domain_name=loader_name,
            domain_role=role,
        )
        return PhaseSpec(
            name=name,
            phase_type=phase_type,  # type: ignore[arg-type]
            loader_group=LoaderGroupSpec(mode="single", loaders=(loader,)),
            enable_backward=phase_type == "train",
            enable_optimizer_step=phase_type == "train",
            enable_scheduler_step=phase_type == "train" and self.cfg.use_cosine_lr,
            scheduler_step_mode="step" if phase_type == "train" and self.cfg.use_cosine_lr else "none",
            grad_clip_norm=self.cfg.grad_clip_norm if phase_type == "train" else None,
            allow_target_domain=phase_type != "train",
        )

    def _loss_fn(
        self, unit_results: Mapping[str, ForwardResult], batch_pack: BatchPack, ctx: StepContext
    ) -> LossResult:
        del batch_pack
        result = unit_results["main"]
        losses = self.loss_module(
            result.loss_inputs["outputs"],
            result.loss_inputs["target"],
            epoch=int(ctx.epoch),
        )
        total = losses["total"]
        assert isinstance(total, torch.Tensor)
        scalars = {"loss": float(total.detach().cpu())}
        components: dict[str, Any] = {}
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                components[key] = value.detach()
                if key != "total":
                    scalars[f"loss_{key}"] = float(value.detach().cpu())
            else:
                components[key] = value
                scalars[key] = float(value)
        return LossResult(total_loss=total, scalars=scalars, components=components)

    def build_trainer(
        self,
        dataset_name: Optional[str] = None,
        train_subset: Optional[str] = None,
        test_subsets: Optional[Sequence[str]] = None,
    ) -> TrainerBuildResult:
        dataset_name = dataset_name or self.cfg.data.dataset_name
        train_subset = train_subset or self.cfg.data.train_subset
        test_subsets = tuple(test_subsets or self.cfg.data.test_subsets)
        fd_ds = self.build_dataset(dataset_name, train_subset)
        train_ds, val_ds = fd_ds["train"], fd_ds["val"]
        model = MWKGNet(
            num_classes=int(train_ds.num_classes),
            in_channels=len(self.cfg.data.signal_keys),
            base_channels=self.cfg.base_channels,
            deep_channels=self.cfg.deep_channels,
            wavelet_levels=self.cfg.wavelet_levels,
            wavelet_kernel_size=self.cfg.wavelet_kernel_size,
            shallow_blocks=self.cfg.shallow_blocks,
            deep_blocks=self.cfg.deep_blocks,
            kgfr_temperature=self.cfg.kgfr_temperature,
            name=self.cfg.model_name,
        )
        phases = [
            self._phase(
                "train",
                "train",
                f"{dataset_name}_{train_subset}_train",
                self._loader_factory(train_ds, True),
                "source",
            ),
            self._phase(
                "val",
                "val",
                f"{dataset_name}_{train_subset}_val",
                self._loader_factory(val_ds, False),
                "source",
            ),
        ]
        for subset in test_subsets:
            if subset == train_subset:
                test_ds = fd_ds["test"]
                role = "source"
            else:
                # Target conditions are evaluation-only. Build a single 100%
                # split so ds2/ds3 are not silently reduced to 15% of samples.
                target_ds = self.build_dataset(
                    dataset_name,
                    subset,
                    split_ratios=(1.0,),
                    split_names=("test",),
                    split_mode="ordered",
                )
                test_ds = target_ds["test"]
                role = "target"
            loader = make_loader(
                test_ds,
                self.cfg.data.batch_size,
                False,
                self.cfg.data.num_workers,
            )
            loader_name = f"{dataset_name}_{subset}"
            phases.append(
                self._phase(
                    f"test_{loader_name}",
                    "test",
                    loader_name,
                    lambda loader=loader: loader,
                    role,
                )
            )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        train_steps = max(1, math.ceil(len(train_ds) / self.cfg.data.batch_size))
        total_steps = max(1, self.cfg.epochs * train_steps)

        def lr_factor(step: int) -> float:
            progress = min(1.0, float(step) / total_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.cfg.min_lr_ratio + (1.0 - self.cfg.min_lr_ratio) * cosine

        scheduler = (
            torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
            if self.cfg.use_cosine_lr
            else None
        )
        root_dir = Path(self.cfg.save_dir) / self.cfg.task_prefix / model.name
        noise_tag = "clean" if self.cfg.data.snr_db is None else f"snr_{self.cfg.data.snr_db:g}db"
        config = RunConfig(
            experiment_name=f"{model.name}_{dataset_name}_{train_subset}",
            run_tag=noise_tag,
            max_epochs=self.cfg.epochs,
            seed=self.cfg.seed,
            device=self.cfg.device,
            phases=tuple(phases),
            path=PathPolicy(root_dir=str(root_dir)),
            async_policy=AsyncPolicy(
                backend="thread" if self.cfg.checkpoint_async else "sync",
                max_workers=2,
                queue_size=8,
                backpressure="block",
            ),
            cache=CachePolicy(metric_prediction_max_samples=50_000),
            checkpoint=CheckpointPolicy(
                save_policy="best_and_last",
                monitor="val.classification.acc",
                monitor_mode="max",
                save_every_n_epochs=max(1, self.cfg.checkpoint_every_n_epochs),
                async_save=self.cfg.checkpoint_async,
                keep_one_per_tag=True,
                save_optimizer=False,
            ),
            log=LogPolicy(
                save_csv=False,
                save_excel=True,
                excel_every_n_epochs=max(1, self.cfg.excel_every_n_epochs),
                excel_notes=(
                    f"method={model.name}; dataset={dataset_name}; train_subset={train_subset}; "
                    f"test_subsets={list(test_subsets)}; snr_db={self.cfg.data.snr_db}; "
                    f"epochs={self.cfg.epochs}; batch_size={self.cfg.data.batch_size}; "
                    f"lr={self.cfg.lr}; wavelet=haar; levels={self.cfg.wavelet_levels}; "
                    f"kurtosis_gamma={self.cfg.kurtosis_gamma}; "
                    "target subsets are evaluated with 100% of their windows",
                ),
                terminal_print_max_metrics=0,
                terminal_metric_include=_metric_patterns(self.cfg.terminal_metric_include),
                terminal_metric_exclude=_metric_patterns(self.cfg.terminal_metric_exclude),
                excel_metric_include=_metric_patterns(self.cfg.excel_metric_include),
                excel_metric_exclude=_metric_patterns(self.cfg.excel_metric_exclude),
                plot_on_end=True,
                plot_every_n_epochs=self.cfg.curve_every_n_epochs,
                curve_model_name=model.name,
                curve_dataset_name=f"{dataset_name}_{train_subset}_{noise_tag}",
            ),
            artifact=ArtifactPolicy(enabled=False),
            analysis=AnalysisPolicy(enabled=False, run_on_end=False),
        )
        unit = ModelUnit(
            name="main",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            forward_adapter=ForwardAdapter(
                x_key=self.cfg.data.x_key,
                y_key=self.cfg.data.y_key,
            ),
        )
        self.build = TrainerBuilder(config).build(
            units={"main": unit},
            loss_fn=self._loss_fn,
            metric_fns={"classification": metric_fn},
        )
        return self.build

    def train(
        self,
        dataset_name: Optional[str] = None,
        train_subset: Optional[str] = None,
        test_subsets: Optional[Sequence[str]] = None,
    ) -> TrainerBuildResult:
        build = self.build_trainer(dataset_name, train_subset, test_subsets)
        build.engine.fit()
        print(f"Run key: {build.paths.run_id}")
        print(f"Metrics XLSX: {build.paths.excel_file('metrics')}")
        print(f"Models: {build.paths.model_dir}")
        return build


def train_mwkg_net(cfg: Optional[TrainConfig] = None) -> TrainerBuildResult:
    return MWKGNetTrainer(cfg=cfg).train()


if __name__ == "__main__":
    train_mwkg_net()
