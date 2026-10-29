# -*- coding: utf-8 -*-
"""Training entry for the paper-faithful IMAG-Net reconstruction.

Published settings retained here:
* simulated dataset: 2,000 signals, five classes, 4,096 points, 12.8 kHz;
* Table-1 network shapes, eight Gabor filters and four MSC iterations;
* the sole training objective is cross entropy (paper Eq. 27).

The default repository experiment uses JNU: ds1 (600 rpm) is the source
condition, while ds2 (800 rpm) and ds3 (1000 rpm) are full target-condition
test sets.  Target-condition samples are never used for optimization.

The paper does not publish optimizer, learning rate, batch size, epoch count,
kernel lengths, or numerical ADMM initial values.  Such values are explicitly
marked as reproduction choices in TrainConfig rather than paper parameters.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import FDPreprocess, FDTrainDataManager
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
    from .model import IMAGNet
except ImportError:
    from model import IMAGNet


CONFIG_PATH = str(PROJECT_ROOT / "configs" / "bearing_datasets_config_strict.yaml")


@dataclass
class SimulationConfig:
    """Paper Eq. (33)-(34) and Table 2."""

    samples: int = 2000
    signal_length: int = 4096
    sampling_rate: float = 12_800.0
    resonance_frequency: float = 3700.0
    shaft_frequency: float = 84.0
    modulation_amplitude: float = 0.3
    impact_mean: float = 1.0
    impact_std: float = 0.1
    damping: float = 0.005
    noise_sigma: float = 1.0
    inner_frequency: float = 454.9
    outer_frequency: float = 301.1
    ball_frequency: float = 197.8
    cage_frequency: float = 42.0


@dataclass
class DataConfig:
    batch_size: int = 64  # reproduction choice; not reported by the paper
    num_workers: int = 0
    signal_key: str = "x"
    label_key: str = "y"
    dataset_name: str = "PU"
    train_subset: str = "ds1"
    test_subsets: Sequence[str] = ("ds1", "ds2", "ds3", "ds4")
    window_size: int = 4096
    overlap: float = 0.0
    split_ratios: Sequence[float] = (0.7, 0.15, 0.15)
    split_names: Sequence[str] = ("train", "val", "test")
    train_ratio: float = 0.8
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    config_path: str = CONFIG_PATH
    build_backend: str = "thread"
    build_workers: int = 4
    loader_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    # The following optimizer settings are reproduction choices because the
    # paper only reports Python 3.11, PyTorch 2.1.0, Windows 11 and RTX 4070Ti.
    epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"
    grad_clip_norm: Optional[float] = 5.0

    # Paper-derived architecture values (Table 1 and Figs. 9-10).
    num_gabor_filters: int = 8
    num_atoms: int = 64
    dictionary_stride: int = 4
    unrolling_steps: int = 4
    adaptive_pool_size: int = 7

    # Unreported implementation values; model.py documents their rationale.
    gabor_kernel_size: int = 129
    # 49, together with classifier kernel 11, reproduces Table 4's 1.50 M size.
    dictionary_kernel_size: int = 49
    classifier_kernel_size: int = 11
    learnable_admm: bool = True
    eta: float = 0.5
    eta2: float = 0.05
    lambda1: float = 1.0
    lambda2: float = 0.01
    lambda3: float = 0.01
    rho: float = 1.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "F:/Python/a01-papers/Paper1.1-pu/"
    task_prefix: str = "paper_reproductions"
    model_name: str = "imag_net"
    seed: int = 3447
    excel_every_n_epochs: int = 100
    curve_every_n_epochs: Optional[int] = None
    checkpoint_async: bool = False
    terminal_metric_include: Sequence[str] = (
        "*.loss",
        "*.classification.acc",
        "*.classification.f1_macro",
        "*.classification.mcc",
    )
    terminal_metric_exclude: Sequence[str] = (
        "*.classification.class_*",
        "*.classification.label_*",
    )
    data: DataConfig = field(default_factory=DataConfig)


class IMAGSimulationDataset(Dataset):
    """Deterministic implementation of the paper's simulated five-class data.

    The paper does not separately define the healthy-class waveform; it is
    represented as the Eq. (33) background-noise term with no fault impacts.
    """

    class_names = ("healthy", "inner", "outer", "ball", "outer_ball_mixed")

    def __init__(
        self,
        config: SimulationConfig,
        indices: Sequence[int],
        *,
        seed: int,
    ) -> None:
        self.config = config
        self.indices = tuple(int(i) for i in indices)
        self.seed = int(seed)
        self.num_classes = 5
        if config.samples % self.num_classes != 0:
            raise ValueError("simulation sample count must be divisible by five classes")

    def __len__(self) -> int:
        return len(self.indices)

    def _impulses(self, rng: np.random.Generator, fault_frequency: float) -> np.ndarray:
        cfg = self.config
        duration = cfg.signal_length / cfg.sampling_rate
        period = 1.0 / fault_frequency
        phase = rng.uniform(0.0, period / 4.0)
        impact_times = np.arange(phase, duration, period)
        amplitudes = rng.normal(cfg.impact_mean, cfg.impact_std, impact_times.size)

        # The response becomes negligible after roughly 10 ms for Table-2 damping.
        response_length = min(cfg.signal_length, int(math.ceil(0.01 * cfg.sampling_rate)))
        local_t = np.arange(response_length, dtype=np.float64) / cfg.sampling_rate
        decay = math.sqrt(cfg.damping / (1.0 - cfg.damping**2))
        response = np.exp(-decay * (2.0 * math.pi * cfg.resonance_frequency * local_t))
        response *= np.cos(2.0 * math.pi * cfg.resonance_frequency * local_t)

        signal = np.zeros(cfg.signal_length, dtype=np.float64)
        for impact_time, amplitude in zip(impact_times, amplitudes):
            start = int(round(impact_time * cfg.sampling_rate))
            if start >= cfg.signal_length:
                continue
            count = min(response_length, cfg.signal_length - start)
            signal[start : start + count] += amplitude * response[:count]
        return signal

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        cfg = self.config
        global_index = self.indices[item]
        per_class = cfg.samples // self.num_classes
        label = min(global_index // per_class, self.num_classes - 1)
        rng = np.random.default_rng(self.seed + global_index)
        signal = np.zeros(cfg.signal_length, dtype=np.float64)
        if label == 1:
            signal += self._impulses(rng, cfg.inner_frequency)
        elif label == 2:
            signal += self._impulses(rng, cfg.outer_frequency)
        elif label == 3:
            signal += self._impulses(rng, cfg.ball_frequency)
        elif label == 4:
            signal += self._impulses(rng, cfg.outer_frequency)
            signal += self._impulses(rng, cfg.ball_frequency)

        time = np.arange(cfg.signal_length, dtype=np.float64) / cfg.sampling_rate
        modulation = 1.0 + cfg.modulation_amplitude * np.cos(
            2.0 * math.pi * cfg.shaft_frequency * time
        )
        signal *= modulation
        signal += rng.normal(0.0, cfg.noise_sigma, cfg.signal_length)
        return {
            "x": torch.from_numpy(signal.astype(np.float32, copy=False)).unsqueeze(0),
            "y": torch.tensor(label, dtype=torch.long),
        }


def build_simulation_splits(
    config: SimulationConfig,
    train_ratio: float,
    seed: int,
) -> tuple[IMAGSimulationDataset, IMAGSimulationDataset]:
    """Class-balanced ordered 80/20 split matching the paper's split ratio."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0,1)")
    per_class = config.samples // 5
    train_count = int(round(per_class * train_ratio))
    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in range(5):
        start = label * per_class
        train_indices.extend(range(start, start + train_count))
        test_indices.extend(range(start + train_count, start + per_class))
    return (
        IMAGSimulationDataset(config, train_indices, seed=seed),
        IMAGSimulationDataset(config, test_indices, seed=seed),
    )


def make_loader(ds: Any, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    generator = torch.Generator().manual_seed(3407)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        generator=generator if shuffle else None,
    )


def _metric_patterns(value: Sequence[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value if str(item).strip())


@dataclass
class ForwardAdapter:
    x_key: str = "x"
    y_key: str = "y"

    def build_inputs(self, unit: ModelUnit, batch_pack: BatchPack, ctx: StepContext) -> ForwardInputs:
        del ctx
        xs: list[Tensor] = []
        ys: list[Tensor] = []
        for view in batch_pack.batches.values():
            xs.append(view.data[self.x_key])
            ys.append(view.data[self.y_key])
        return ForwardInputs(
            args=(torch.cat(xs, dim=0).to(unit.device),),
            kwargs={"return_intermediates": False},
            meta={"target": torch.cat(ys, dim=0).long().to(unit.device)},
        )

    def forward(self, unit: ModelUnit, inputs: ForwardInputs, ctx: StepContext) -> ForwardResult:
        del ctx
        outputs = unit.model(*inputs.args, **inputs.kwargs)
        target = inputs.meta["target"]
        return ForwardResult(
            main=outputs["logits"],
            loss_inputs={"logits": outputs["logits"], "target": target},
            metric_inputs={"logits": outputs["logits"], "target": target},
            features={"features": outputs["features"]},
            attentions={"center_frequencies": outputs["center_frequencies"]},
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


class IMAGNetTrainer:
    def __init__(self, cfg: Optional[TrainConfig] = None):
        self.cfg = cfg or TrainConfig()
        data = self.cfg.data
        # if data.dataset_name != "JNU":
        #     raise ValueError("This training entry is configured for the JNU dataset")
        if data.window_size != 4096:
            raise ValueError("IMAG-Net Table 1 requires window_size=4096")
        if data.window_size % self.cfg.dictionary_stride != 0:
            raise ValueError("window_size must be divisible by dictionary_stride")
        if not data.test_subsets:
            raise ValueError(f"test_subsets must contain at least one {cfg.data.dataset_name} condition")
        if len(set(data.test_subsets)) != len(data.test_subsets):
            raise ValueError("test_subsets must not contain duplicates")
        self.fd_manager = FDTrainDataManager(config=data.config_path)
        self.build: Optional[TrainerBuildResult] = None

    def build_repository_dataset(
        self,
        subset_name: str,
        *,
        split_ratios: Sequence[float],
        split_names: Sequence[str],
        split_mode: str,
    ) -> Dict[str, Any]:
        """Build one JNU operating condition with the requested split policy."""
        data = self.cfg.data
        return self.fd_manager.build_and_split_and_to_torch_dataset(
            dataset_name=data.dataset_name,
            subset_name=[subset_name],
            loader_kwargs=dict(data.loader_kwargs),
            post_parse_pipeline=[
                {
                    "level": "dataset",
                    "fns": [
                        FDPreprocess.dataset_sliding_window(
                            window_size=data.window_size,
                            overlap=data.overlap,
                            axis=0,
                            keys=[data.signal_key],
                            drop_last=True,
                        )
                    ],
                },
                {
                    "level": "sample",
                    "fns": [FDPreprocess.sample_expand_dims(axis=0, keys=[data.signal_key])],
                },
            ],
            backend=data.build_backend,
            num_workers=data.build_workers,
            split_ratios=tuple(split_ratios),
            split_names=tuple(split_names),
            split_mode=split_mode,
            split_group_by="label",
            shuffle_within_split=False,
            seed=self.cfg.seed,
            x_key=data.signal_key,
            y_key=data.label_key,
            return_dict=True,
        )

    def _phase(
        self,
        name: str,
        phase_type: str,
        loader: DataLoader,
        *,
        role: str,
    ) -> PhaseSpec:
        spec = LoaderSpec(
            name=name,
            loader_factory=lambda loader=loader: loader,
            domain_name=name,
            domain_role=role,
        )
        return PhaseSpec(
            name=name,
            phase_type=phase_type,  # type: ignore[arg-type]
            loader_group=LoaderGroupSpec(mode="single", loaders=(spec,)),
            enable_backward=phase_type == "train",
            enable_optimizer_step=phase_type == "train",
            enable_scheduler_step=False,
            scheduler_step_mode="none",
            grad_clip_norm=self.cfg.grad_clip_norm if phase_type == "train" else None,
            allow_target_domain=phase_type != "train",
        )

    @staticmethod
    def _loss_fn(
        unit_results: Mapping[str, ForwardResult], batch_pack: BatchPack, ctx: StepContext
    ) -> LossResult:
        del batch_pack, ctx
        inputs = unit_results["main"].loss_inputs
        # Paper Eq. (27): no auxiliary reconstruction or sparsity loss is added.
        total = F.cross_entropy(inputs["logits"], inputs["target"])
        return LossResult(
            total_loss=total,
            scalars={"loss": float(total.detach().cpu())},
            components={"cross_entropy": total.detach()},
        )

    def build_trainer(self, *, dataset_mode: str = "jnu") -> TrainerBuildResult:
        if dataset_mode == "simulation":
            train_ds, test_ds = build_simulation_splits(
                self.cfg.data.simulation,
                self.cfg.data.train_ratio,
                self.cfg.seed,
            )
            dataset_tag = f"simulation_sigma_{self.cfg.data.simulation.noise_sigma}"
            datasets = {"train": train_ds, "test": test_ds}
        else: #
            data = self.cfg.data
            datasets = self.build_repository_dataset(
                data.train_subset,
                split_ratios=data.split_ratios,
                split_names=data.split_names,
                split_mode="random",
            )
            train_ds = datasets["train"]
            test_ds = datasets["test"]
            dataset_tag = f"{data.dataset_name}_{data.train_subset}_to_{'_'.join(data.test_subsets)}"


        num_classes = int(train_ds.num_classes)
        model = IMAGNet(
            num_classes=num_classes,
            num_gabor_filters=self.cfg.num_gabor_filters,
            gabor_kernel_size=self.cfg.gabor_kernel_size,
            num_atoms=self.cfg.num_atoms,
            dictionary_kernel_size=self.cfg.dictionary_kernel_size,
            dictionary_stride=self.cfg.dictionary_stride,
            unrolling_steps=self.cfg.unrolling_steps,
            classifier_kernel_size=self.cfg.classifier_kernel_size,
            adaptive_pool_size=self.cfg.adaptive_pool_size,
            learnable_admm=self.cfg.learnable_admm,
            eta=self.cfg.eta,
            eta2=self.cfg.eta2,
            lambda1=self.cfg.lambda1,
            lambda2=self.cfg.lambda2,
            lambda3=self.cfg.lambda3,
            rho=self.cfg.rho,
            name=self.cfg.model_name,
        )
        if self.cfg.optimizer.lower() != "adam":
            raise ValueError("Only the configured Adam reproduction choice is supported")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        data = self.cfg.data
        train_loader = make_loader(train_ds, data.batch_size, True, data.num_workers)
        phases: list[PhaseSpec] = [
            self._phase("train", "train", train_loader, role="source")
        ]
        if dataset_mode == "jnu":
            val_loader = make_loader(datasets["val"], data.batch_size, False, data.num_workers)
            phases.append(self._phase("val", "val", val_loader, role="source"))
            for subset in data.test_subsets:
                if subset == data.train_subset:
                    subset_test = test_ds
                    role = "source"
                else:
                    # Target conditions are evaluation-only and are not split:
                    # every available 4096-point window is tested.
                    target = self.build_repository_dataset(
                        subset,
                        split_ratios=(1.0,),
                        split_names=("test",),
                        split_mode="ordered",
                    )
                    subset_test = target["test"]
                    role = "target"
                loader = make_loader(subset_test, data.batch_size, False, data.num_workers)
                phase_name = f"test_{data.dataset_name}_{subset}"
                phases.append(self._phase(phase_name, "test", loader, role=role))
        else:
            test_loader = make_loader(test_ds, data.batch_size, False, data.num_workers)
            phases.append(self._phase("test", "test", test_loader, role="source"))
        root_dir = Path(self.cfg.save_dir) / self.cfg.task_prefix / model.name
        config = RunConfig(
            experiment_name=f"{model.name}_{dataset_tag}",
            run_tag=dataset_tag,
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
                save_last_every_n_epochs=None,
                async_save=self.cfg.checkpoint_async,
                keep_one_per_tag=True,
                save_optimizer=False,
            ),
            log=LogPolicy(
                save_csv=False,
                save_excel=True,
                excel_every_n_epochs=max(1, self.cfg.excel_every_n_epochs),
                excel_notes=(
                    f"method=IMAG-Net; source=Cao et al. MSSP 257 (2026) 114586; "
                    f"dataset={dataset_tag}; loss=cross_entropy_only; "
                    f"filters={self.cfg.num_gabor_filters}; atoms={self.cfg.num_atoms}; "
                    f"unrolling_steps={self.cfg.unrolling_steps}; input_length=4096; "
                    f"parameters={model.parameter_count()}; "
                    f"train_subset={data.train_subset}; test_subsets={list(data.test_subsets)}; "
                    f"{data.dataset_name} target subsets are evaluation-only and use 100% of windows; "
                    "optimizer/lr/batch/epochs and kernel lengths are unreported reproduction choices"
                ),
                terminal_print_max_metrics=0,
                terminal_metric_include=_metric_patterns(self.cfg.terminal_metric_include),
                terminal_metric_exclude=_metric_patterns(self.cfg.terminal_metric_exclude),
                plot_on_end=True,
                plot_every_n_epochs=self.cfg.curve_every_n_epochs,
                curve_model_name=model.name,
                # Keep generated Windows filenames below MAX_PATH. Full source
                # and target condition details remain in experiment_name/notes.
                curve_dataset_name=data.dataset_name,
            ),
            artifact=ArtifactPolicy(enabled=False),
            analysis=AnalysisPolicy(enabled=False, run_on_end=False),
        )
        unit = ModelUnit(
            name="main",
            model=model,
            optimizer=optimizer,
            scheduler=None,
            forward_adapter=ForwardAdapter(
                x_key=self.cfg.data.signal_key,
                y_key=self.cfg.data.label_key,
            ),
        )
        self.build = TrainerBuilder(config).build(
            units={"main": unit},
            loss_fn=self._loss_fn,
            metric_fns={"classification": metric_fn},
        )
        return self.build

    def train(self, *, dataset_mode: str = "jnu") -> TrainerBuildResult:
        build = self.build_trainer(dataset_mode=dataset_mode)
        build.engine.fit()
        print(f"Run key: {build.paths.run_id}")
        print(f"Metrics XLSX: {build.paths.excel_file('metrics')}")
        print(f"Models: {build.paths.model_dir}")
        return build


def train_imag_net(
    cfg: Optional[TrainConfig] = None,
    *,
    dataset_mode: str = "jnu",
) -> TrainerBuildResult:
    return IMAGNetTrainer(cfg=cfg).train(dataset_mode=dataset_mode)


if __name__ == "__main__":
    train_imag_net()


__all__ = [
    "SimulationConfig",
    "DataConfig",
    "TrainConfig",
    "IMAGSimulationDataset",
    "build_simulation_splits",
    "IMAGNetTrainer",
    "train_imag_net",
]
