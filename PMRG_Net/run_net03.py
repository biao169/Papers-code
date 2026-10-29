# -*- coding: utf-8 -*-
"""Motor bearing fault diagnosis from vibration signals under source-only training.

The model is designed for condition generalization: it is trained on one working
condition and evaluated on other speed/load conditions.  The core front-end is a
multi-scale Gabor envelope transform with log-spaced center frequencies and
Q-factor width parameterization, so spacing and Q-factor losses are no longer
needed.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import FDPreprocess, FDTrainDataManager
from dl_framework import (
    AnalysisPolicy,
    ArtifactPolicy,
    AsyncPolicy,
    BatchPack,
    CachePolicy,
    CheckpointPolicy,
    FinalAnalysisPolicy,
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
from dl_framework.services.model_analysis import AnalysisData, ModelAnalysisPolicy, UnifiedModelAnalyzer
from dl_framework.services.xai import XAIPolicy, XAIRequest, UnifiedXAIAnalyzer

try:
    from .physics_net import GaborGeneralizationLoss, MotorBearingGaborNet, VibrationNetConfig
except ImportError:
    from physics_net import GaborGeneralizationLoss, MotorBearingGaborNet, VibrationNetConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = str(PROJECT_ROOT / "configs" / "bearing_datasets_config_strict.yaml")
SEED = [3608, 3618, 3628, 3638, 3648, 3658, 3668, 3678, ]

@dataclass(frozen=True)
class ExperimentSpec:
    """One experiment setting; name is used as the model and output folder name."""

    name: str
    description: str
    front_end: str = "multiscale_gabor"
    spacing_mode: str = "log"
    quadrature: bool = False
    learnable_gabor: bool = True
    use_se: bool = True
    normalize_embedding: bool = False
    augment: bool = True
    filters_per_scale: Sequence[int] = (8, 8, 8)
    kernel_sizes: Sequence[int] = (65, 129, 257)
    use_physics_loss: bool = True
    lambda_decor: float = 1e-3
    lambda_impulse: float = 1e-4
    label_smoothing: float = 0.05


@dataclass
class DataConfig:
    window_size: int = 4096
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
    lr: float = 8e-4 #8e-4  #
    weight_decay: float = 5e-4
    emb_dim: int = 512
    fs_hz: Optional[float] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "F:/Python/a01-papers/Paper1.1-pu/"
    task_prefix: str = "net03_generalization"
    plot_enabled: bool = True
    curve_every_n_epochs: int = 1000
    excel_every_n_epochs: int = 5000
    checkpoint_every_n_epochs: int = 1000
    checkpoint_async: bool = False
    terminal_metric_include: Sequence[str] = (
        "*.loss",
        "*.classification.acc",
        "*.classification.precision_macro",
        "*.classification.recall_macro",
        "*.classification.f1_macro",
        "*.classification.mcc",
    )
    terminal_metric_exclude: Sequence[str] = ("*.classification.class_*", "*.classification.label_*")
    excel_metric_include: Sequence[str] = (
        "*.loss",
        "*.classification.acc",
        "*.classification.precision_macro",
        "*.classification.recall_macro",
        "*.classification.f1_macro",
        "*.classification.mcc",
    )
    excel_metric_exclude: Sequence[str] = ("*.classification.class_*", "*.classification.label_*")
    artifact_keep_every_n_steps: int = 1
    analysis_enabled: bool = True
    analysis_max_samples: int = 5000
    xai_enabled: bool = True
    xai_max_samples: int = 3
    xai_methods: Sequence[str] = (
        "input_attribution_1d",
        "integrated_gradients_1d",
        "smoothgrad_1d",
        "grad_cam_1d",
        "frequency_importance",
    )
    data: DataConfig = field(default_factory=DataConfig)


def _float_tag(value: float) -> str:
    if float(value) == 0.0:
        return "0"
    return f"{float(value):.0e}".replace("+", "").replace("-", "m")


def _metric_patterns(value: Sequence[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(x) for x in value if str(x).strip())


def make_final_spec(**kwargs: Any) -> ExperimentSpec:
    """Final Gabor model for source-only condition generalization."""

    base = ExperimentSpec(
        name="msgabor_condition_generalization",
        description="Multi-scale log-spaced Q-parameterized Gabor vibration model.",
    )
    return replace(base, **kwargs) if kwargs else base


LOSS_SWEEP_VALUES: dict[str, tuple[float, ...]] = {
    "lambda_decor": (0.0, 1e-4, 1e-3, 1e-2, 5e-2),
    "lambda_impulse": (0.0, 1e-4, 1e-3, 5e-3, 1e-2),
}


def make_loss_ablation_specs(
    sweep_values: Mapping[str, Sequence[float]] | None = None,
    *,
    include_ce_only: bool = True,
) -> tuple[ExperimentSpec, ...]:
    """Loss ablation around the final Gabor structure."""

    base = make_final_spec(quadrature=False)
    specs: list[ExperimentSpec] = []
    if include_ce_only:
        specs.append(replace(base, name="loss_ce_only", description="Classification loss only.", use_physics_loss=False, lambda_decor=0.0, lambda_impulse=0.0))
    for loss_name, values in (sweep_values or LOSS_SWEEP_VALUES).items():
        for value in values:
            short = loss_name.replace("lambda_", "")
            specs.append(replace(base, name=f"loss_{short}_{_float_tag(float(value))}", description=f"{loss_name}={float(value):g}.", **{loss_name: float(value)}))
    return tuple(specs)


def make_structure_ablation_specs(
    *,
    lambda_decor: float = 1e-2,
    lambda_impulse: float = 1e-3,
    label_smoothing: float = 0.05,
) -> tuple[ExperimentSpec, ...]:
    """Gabor-only structure ablation, no more than four settings."""

    final = make_final_spec(lambda_decor=lambda_decor, lambda_impulse=lambda_impulse, label_smoothing=label_smoothing)
    return (
        replace(final, name="struct_msgabor_log_q", description="Final multi-scale log-spaced Q-parameterized Gabor.", quadrature=True),
        replace(final, name="struct_linear_spacing", description="Linear center-frequency spacing.", spacing_mode="linear"),
        replace(final, name="struct_single_scale_gabor", description="Single-scale Gabor front-end.", front_end="single_gabor"),
        replace(final, name="struct_no_quadrature", description="Real Gabor magnitude without quadrature envelope.", quadrature=False),
    )


def make_loss_ablation_config(cfg: Optional[TrainConfig] = None) -> TrainConfig:
    run_cfg = copy.deepcopy(cfg) if cfg is not None else TrainConfig()
    run_cfg.task_prefix = "net03_loss_ablation"
    run_cfg.analysis_enabled = False
    run_cfg.xai_enabled = False
    return run_cfg


def make_structure_ablation_config(cfg: Optional[TrainConfig] = None) -> TrainConfig:
    run_cfg = copy.deepcopy(cfg) if cfg is not None else TrainConfig()
    run_cfg.task_prefix = f"net03_structure_ablation"   # -lr{run_cfg.lr:.4f}({run_cfg.weight_decay:.4f})
    return run_cfg


def make_sci_ablation_specs() -> tuple[ExperimentSpec, ...]:
    final = make_final_spec(name="msgabor_full", description="Final Gabor condition-generalization model.")
    return (
        final,
        replace(final, name="msgabor_ce_only", description="No decorrelation or impulse prior.", use_physics_loss=False, lambda_decor=0.0, lambda_impulse=0.0),
        replace(final, name="msgabor_no_decor", description="Remove scale decorrelation prior.", lambda_decor=0.0),
        replace(final, name="msgabor_no_impulse", description="Remove impulse prior.", lambda_impulse=0.0),
        replace(final, name="msgabor_linear_spacing", description="Use linear Gabor center spacing.", spacing_mode="linear"),
        replace(final, name="msgabor_single_scale", description="Use a single Gabor scale.", front_end="single_gabor"),
        replace(final, name="msgabor_no_quadrature", description="Disable quadrature envelope.", quadrature=False),
    )


@dataclass
class VibrationForwardAdapter:
    x_key: str = "x"
    y_key: str = "y"

    def build_inputs(self, unit: ModelUnit, batch_pack: BatchPack, ctx: StepContext) -> ForwardInputs:
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        domains: list[str] = []
        for view in batch_pack.batches.values():
            batch = view.data
            xs.append(batch[self.x_key])
            ys.append(batch[self.y_key])
            domains.extend([view.domain_name] * int(batch[self.y_key].shape[0]))
        x = torch.cat(xs, dim=0).to(unit.device)
        y = torch.cat(ys, dim=0).long().to(unit.device)
        return ForwardInputs(args=(x,), kwargs={"return_feature_map": getattr(ctx, "phase_type", "") != "train"}, meta={"target": y, "domains": domains})

    def forward(self, unit: ModelUnit, inputs: ForwardInputs, ctx: StepContext) -> ForwardResult:
        outputs = unit.model(*inputs.args, **inputs.kwargs)
        logits = outputs["logits"]
        target = inputs.meta["target"]
        keep_features = getattr(ctx, "phase_type", "") != "train"
        return ForwardResult(
            main=logits,
            loss_inputs={"outputs": outputs, "logits": logits, "target": target},
            metric_inputs={"logits": logits, "target": target},
            features={"features": outputs.get("features"), "feature_map": outputs.get("feature_map")} if keep_features else None,
        )


def vibration_loss_fn(
    unit_results: Mapping[str, ForwardResult],
    batch_pack: BatchPack,
    ctx: StepContext,
    loss_module: Optional[nn.Module] = None,
) -> LossResult:
    del batch_pack
    result = unit_results["main"]
    target = result.loss_inputs["target"]
    if loss_module is None:
        total_loss = F.cross_entropy(result.loss_inputs["logits"], target, label_smoothing=0.05)
        scalars = {"cls": float(total_loss.detach().cpu())}
    else:
        loss_dict = loss_module(result.loss_inputs["outputs"], target, epoch=ctx.epoch)
        total_loss = loss_dict.get("total", loss_dict.get("loss"))
        if total_loss is None:
            total_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
        scalars = {k: float(v.detach().cpu()) for k, v in loss_dict.items() if isinstance(v, torch.Tensor) and v.numel() == 1}
    scalars["loss"] = float(total_loss.detach().cpu())
    return LossResult(total_loss=total_loss, scalars=scalars)


def vibration_metrics_fn(unit_results: Mapping[str, ForwardResult], batch_pack: BatchPack, ctx: StepContext) -> dict[str, Any]:
    del batch_pack, ctx
    result = unit_results["main"]
    return {"classification": {"y_true": result.metric_inputs["target"], "y_score": result.metric_inputs["logits"], "task": "single_label"}}


def make_loader(ds: Any, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers)


class MotorBearingDiagnosisRunner:
    """Training runner for source-only motor bearing condition generalization."""

    def __init__(self, config_path: str = CONFIG_PATH, cfg: Optional[TrainConfig] = None) -> None:
        self.config_path = config_path
        self.cfg = cfg or TrainConfig()
        self.fd_manager = FDTrainDataManager(config=config_path)
        self.build: Optional[TrainerBuildResult] = None
        self.loss_module: Optional[nn.Module] = None
        self._analysis_loaders: dict[str, DataLoader] = {}
        self._class_names: tuple[str, ...] | None = None

    def build_dataset(self, dataset_name: str = "PU", subset_name: Sequence[str] | str = "ds1") -> Dict[str, Any]:
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
            split_mode="ordered",
            split_group_by="label",
            shuffle_within_split=False,
            x_key=self.cfg.data.x_key,
            y_key=self.cfg.data.y_key,
            return_dict=True,
        )

    def build_test_loaders(self, dataset_name: str, subsets: Sequence[str]) -> dict[str, DataLoader]:
        loaders = {}
        for subset in subsets:
            ds = self.build_dataset(dataset_name=dataset_name, subset_name=subset)
            loaders[f"{dataset_name}_{subset}"] = make_loader(ds["test"], self.cfg.data.batch_size, False, self.cfg.data.num_workers)
        return loaders

    def _loader_factory(self, ds: Any, *, shuffle: bool):
        def factory() -> DataLoader:
            return make_loader(ds, self.cfg.data.batch_size, shuffle, self.cfg.data.num_workers)
        return factory

    def _make_phase(self, name: str, phase_type: str, loader_name: str, loader_factory: Any, *, domain_role: str) -> PhaseSpec:
        loader = LoaderSpec(name=loader_name, loader_factory=loader_factory, loader_kwargs={}, domain_name=loader_name, domain_role=domain_role)
        return PhaseSpec(
            name=name,
            phase_type=phase_type,  # type: ignore[arg-type]
            loader_group=LoaderGroupSpec(mode="single", loaders=(loader,)),
            enable_backward=phase_type == "train",
            enable_optimizer_step=phase_type == "train",
            grad_clip_norm=1.0 if phase_type == "train" else None,
            allow_target_domain=phase_type != "train",
        )

    def build_model(self, spec: ExperimentSpec, train_ds: Any, fs_hz: float) -> MotorBearingGaborNet:
        model_cfg = VibrationNetConfig(
            in_channels=len(self.cfg.data.signal_keys),
            num_classes=int(train_ds.num_classes),
            fs_hz=fs_hz,
            emb_dim=self.cfg.emb_dim,
            front_end=spec.front_end,  # type: ignore[arg-type]
            spacing_mode=spec.spacing_mode,  # type: ignore[arg-type]
            quadrature=spec.quadrature,
            learnable_gabor=spec.learnable_gabor,
            use_se=spec.use_se,
            normalize_embedding=spec.normalize_embedding,
            augment=spec.augment,
            filters_per_scale=spec.filters_per_scale,
            kernel_sizes=spec.kernel_sizes,
        )
        return MotorBearingGaborNet(model_cfg, name=spec.name)

    def build_loss(self, spec: ExperimentSpec, model: MotorBearingGaborNet) -> Optional[nn.Module]:
        del model
        if not spec.use_physics_loss:
            return None
        return GaborGeneralizationLoss(lambda_decor=spec.lambda_decor, lambda_impulse=spec.lambda_impulse, label_smoothing=spec.label_smoothing)

    def build_trainer(
        self,
        *,
        spec: ExperimentSpec,
        dataset_name: str = "PU",
        train_subset: str = "ds1",
        test_subsets: Sequence[str] = ("ds1", "ds2", "ds3", "ds4"),
    ) -> TrainerBuildResult:
        fd_ds = self.build_dataset(dataset_name=dataset_name, subset_name=train_subset)
        train_ds, val_ds = fd_ds["train"], fd_ds["val"]
        print(f"{dataset_name}_{train_subset}>> train_ds length: {len(train_ds)}, val_ds length: {len(val_ds)}")
        test_loaders = self.build_test_loaders(dataset_name, test_subsets)
        testset_str = [f"{n} length:{len(d.dataset)}" for n, d in test_loaders.items()]
        print(" \n".join( testset_str))
        # exit()
        fs_hz = float(self.cfg.fs_hz or self.cfg.data.target_fs or getattr(train_ds, "sampling_rate", 1.0) or 1.0)
        self._class_names = tuple(str(i) for i in range(int(train_ds.num_classes)))
        self._analysis_loaders = {"val": make_loader(val_ds, self.cfg.data.batch_size, False, self.cfg.data.num_workers)}
        self._analysis_loaders.update({f"test_{name}": loader for name, loader in test_loaders.items()})

        model = self.build_model(spec, train_ds, fs_hz)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.loss_module = self.build_loss(spec, model)
        phases = [
            self._make_phase("train", "train", f"{dataset_name}_{train_subset}_train", self._loader_factory(train_ds, shuffle=True), domain_role="source"),
            self._make_phase("val", "val", f"{dataset_name}_{train_subset}_val", self._loader_factory(val_ds, shuffle=False), domain_role="source"),
        ]
        for name, loader in test_loaders.items():
            role = "source" if name.split("_")[-1] == train_subset else "target"
            phases.append(self._make_phase(f"test_{name}", "test", name, lambda loader=loader: loader, domain_role=role))

        root_dir = Path(self.cfg.save_dir) / self.cfg.task_prefix / model.name
        config = RunConfig(
            experiment_name=f"{model.name}_{dataset_name}_{train_subset}",
            run_tag="train",
            seed=SEED[0],  # 3608,
            max_epochs=self.cfg.epochs,
            device=self.cfg.device,
            phases=tuple(phases),
            path=PathPolicy(root_dir=str(root_dir)),
            async_policy=AsyncPolicy(backend="thread" if self.cfg.checkpoint_async else "sync", max_workers=2, queue_size=8, backpressure="block"),
            cache=CachePolicy(metric_prediction_max_samples=50_000),
            checkpoint=CheckpointPolicy(save_policy="best_and_last", monitor="val.classification.f1_macro", monitor_mode="max", save_every_n_epochs=max(1, int(self.cfg.checkpoint_every_n_epochs)), async_save=bool(self.cfg.checkpoint_async), keep_one_per_tag=True, save_optimizer=False, save_scheduler=False, save_scaler=False, save_trainer_state=False, save_epoch_record=False),
            log=LogPolicy(
                print_every_phase=True,
                save_csv=False,
                save_excel=True,
                flush_every_n_epochs=1,
                max_buffer_rows=32,
                excel_every_n_epochs=max(1, int(self.cfg.excel_every_n_epochs)),
                excel_notes=(f"model={model.name}; ablation={spec.description}; dataset={dataset_name}; train_subset={train_subset}; test_subsets={list(test_subsets)}; window={self.cfg.data.window_size}; overlap={self.cfg.data.overlap}",),
                terminal_print_max_metrics=0,
                terminal_metric_include=_metric_patterns(self.cfg.terminal_metric_include),
                terminal_metric_exclude=_metric_patterns(self.cfg.terminal_metric_exclude),
                excel_metric_include=_metric_patterns(self.cfg.excel_metric_include),
                excel_metric_exclude=_metric_patterns(self.cfg.excel_metric_exclude),
                plot_on_end=True,
                plot_every_n_epochs=int(self.cfg.curve_every_n_epochs) if self.cfg.plot_enabled else None,
                curve_max_points=5000,
                curve_model_name=model.name,
                curve_dataset_name=f"{dataset_name}_{train_subset}",
            ),
            artifact=ArtifactPolicy(enabled=True, phase_names=("val",), every_n_epochs=max(1, int(self.cfg.artifact_keep_every_n_steps)), max_events_per_epoch=1, max_samples_per_event=4, max_snapshot_files=32),
            analysis=AnalysisPolicy(enabled=False, run_on_end=False),
            final_analysis=FinalAnalysisPolicy(enabled=bool(self.cfg.analysis_enabled or self.cfg.xai_enabled), phase_patterns=("test*",), unit_name="main", task_type="timeseries", model_name=model.name, dataset_name=f"{dataset_name}_{train_subset}", max_samples=max(1, int(self.cfg.analysis_max_samples)), run_model_analysis=bool(self.cfg.analysis_enabled), run_xai=bool(self.cfg.xai_enabled), xai_methods=tuple(self.cfg.xai_methods), xai_max_samples=0, samples_per_label=min(3, max(1, int(self.cfg.xai_max_samples))), save_json=False, save_csv=False, save_xlsx=False, enable_heavy_feature_plots=False),
        )
        unit = ModelUnit(name="main", model=model, optimizer=optimizer, forward_adapter=VibrationForwardAdapter(x_key=self.cfg.data.x_key, y_key=self.cfg.data.y_key))
        build = TrainerBuilder(config).build(units={"main": unit}, loss_fn=self._loss_fn, metric_fns={"classification": vibration_metrics_fn})
        self.build = build
        return build

    def _loss_fn(self, unit_results: Mapping[str, ForwardResult], batch_pack: BatchPack, ctx: StepContext) -> LossResult:
        return vibration_loss_fn(unit_results, batch_pack, ctx, loss_module=self.loss_module)

    def train_one(self, *, spec: ExperimentSpec, dataset_name: str = "PU", train_subset: str = "ds1", test_subsets: Sequence[str] = ("ds1", "ds2", "ds3", "ds4")) -> TrainerBuildResult:
        build = self.build_trainer(spec=spec, dataset_name=dataset_name, train_subset=train_subset, test_subsets=test_subsets)
        records = build.engine.fit()
        final_metrics = records[-1].scalars if records else {}
        self.run_final_analysis(build, final_metrics=final_metrics)
        self.export_gabor_interpretation(build, final_metrics=final_metrics)
        print(f"\nRun key:       {build.paths.run_id}")
        print(f"Model name:    {spec.name}")
        print(f"Metrics XLSX:  {build.paths.excel_file('metrics')}")
        print(f"Figures:       {build.paths.figure_dir}")
        return build

    def train_suite(self, *, specs: Sequence[ExperimentSpec] | None = None, selected_names: Sequence[str] | None = None, dataset_name: str = "PU", train_subset: str = "ds1", test_subsets: Sequence[str] = ("ds1", "ds2", "ds3", "ds4")) -> list[TrainerBuildResult]:
        specs = tuple(specs or make_sci_ablation_specs())
        if selected_names:
            wanted = {str(x) for x in selected_names}
            specs = tuple(s for s in specs if s.name in wanted)
        builds = []
        for idx, spec in enumerate(specs, start=1):
            print(f"\n===== motor bearing experiment {idx}/{len(specs)}: {spec.name} =====")
            builds.append(self.train_one(spec=spec, dataset_name=dataset_name, train_subset=train_subset, test_subsets=test_subsets))
        return builds

    def _target_layer(self, model: nn.Module) -> nn.Module | None:
        convs = [m for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d))]
        return convs[-1] if convs else None

    def _forward_for_xai(self, model: nn.Module, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return model(inputs, return_feature_map=True)

    @staticmethod
    def _analysis_name_parts(name: str) -> tuple[str, str]:
        if name == "val":
            return "val", "validation"
        if name.startswith("test_"):
            return "test", name[len("test_"):]
        return name, name

    def _collect_analysis_data(self, model: nn.Module, loader: DataLoader, *, device: str, max_samples: int) -> dict[str, torch.Tensor | None]:
        ys, preds, logits_list, features, inputs = [], [], [], [], []
        collected = 0
        model.to(device)
        model.eval()
        with torch.no_grad():
            for batch in loader:
                x = batch[self.cfg.data.x_key].to(device)
                y = batch[self.cfg.data.y_key].long().to(device)
                out = model(x, return_feature_map=True)
                logits = out["logits"]
                take = min(int(x.shape[0]), max(0, int(max_samples) - collected))
                if take <= 0:
                    break
                ys.append(y[:take].cpu())
                preds.append(logits[:take].argmax(dim=1).cpu())
                logits_list.append(logits[:take].cpu())
                inputs.append(x[:take].cpu())
                if isinstance(out.get("features"), torch.Tensor):
                    features.append(out["features"][:take].cpu())
                collected += take
                if collected >= max_samples:
                    break
        return {"y_true": torch.cat(ys) if ys else None, "y_pred": torch.cat(preds) if preds else None, "logits": torch.cat(logits_list) if logits_list else None, "features": torch.cat(features) if features else None, "inputs": torch.cat(inputs) if inputs else None}

    def run_final_analysis(self, build: TrainerBuildResult, *, final_metrics: Mapping[str, float] | None = None) -> dict[str, int]:
        counts = {"analysis_artifacts": 0, "xai_artifacts": 0}
        if not self.cfg.analysis_enabled and not self.cfg.xai_enabled:
            return counts
        unit = build.unit_group["main"]
        model = unit.model
        model_name = getattr(model, "name", "PMRG_Net")
        device = build.engine.config.device
        target_layer = self._target_layer(model)
        for analysis_name, loader in self._analysis_loaders.items():
            phase_name, dataset_name = self._analysis_name_parts(analysis_name)
            data = self._collect_analysis_data(model, loader, device=device, max_samples=max(1, int(self.cfg.analysis_max_samples)))
            if data["y_true"] is None or data["logits"] is None:
                continue
            if self.cfg.analysis_enabled:
                analyzer = UnifiedModelAnalyzer(build.paths, ModelAnalysisPolicy(model_name=model_name, dataset_name=dataset_name, max_samples=max(1, int(self.cfg.analysis_max_samples)), save_json=False, save_csv=False, save_xlsx=False, enable_heavy_feature_plots=False))
                bundle = analyzer.analyze(AnalysisData(task_type="timeseries", y_true=data["y_true"], y_pred=data["y_pred"], logits=data["logits"], features=data["features"], inputs=data["inputs"], class_names=self._class_names, metadata={"phase": phase_name, "analysis_metrics": dict(final_metrics or {})}), name=f"{analysis_name}_final")
                counts["analysis_artifacts"] += len(bundle.artifacts)
            if self.cfg.xai_enabled and data["inputs"] is not None:
                xai = UnifiedXAIAnalyzer(build.paths, XAIPolicy(methods=tuple(self.cfg.xai_methods), max_points=5000))
                result = xai.explain(XAIRequest(model=model, inputs=data["inputs"][:1], target=data["y_true"][:1] if data["y_true"] is not None else None, device=device, forward_fn=self._forward_for_xai, target_layer=target_layer, attentions=None, name=f"{analysis_name}_xai", model_name=model_name, dataset_name=dataset_name, metadata={"phase": phase_name, "metrics": dict(final_metrics or {})}))
                counts["xai_artifacts"] += len(result.artifacts)
        model.to(device)
        model.eval()
        return counts

    def export_gabor_interpretation(self, build: TrainerBuildResult, *, final_metrics: Mapping[str, float] | None = None, max_samples: int = 512) -> dict[str, Any]:
        unit = build.unit_group["main"]
        model = unit.model
        model_name = getattr(model, "name", "PMRG_Net")
        if not isinstance(model, MotorBearingGaborNet) or model.gabor is None:
            return {"skipped": "model has no Gabor front-end"}
        loader = self._analysis_loaders.get("val") or next(iter(self._analysis_loaders.values()), None)
        if loader is None:
            return {"skipped": "no analysis loader"}
        device = build.engine.config.device
        model.to(device)
        model.eval()
        class_energy: dict[int, list[torch.Tensor]] = {}
        sample_records: list[dict[str, Any]] = []
        with torch.no_grad():
            seen = 0
            for batch in loader:
                x = batch[self.cfg.data.x_key].to(device)
                y = batch[self.cfg.data.y_key].long().to(device)
                out = model(x, return_feature_map=False)
                pred = out["logits"].argmax(dim=1)
                env = out["envelope"].detach().float().cpu()
                energy = env.mean(dim=(1, 3))
                for idx, (label, pred_label, row) in enumerate(zip(y.cpu().tolist(), pred.cpu().tolist(), energy)):
                    class_energy.setdefault(int(label), []).append(row)
                    if len(sample_records) < 12:
                        sample_records.append({
                            "x": x[idx, 0].detach().float().cpu(),
                            "env": env[idx, 0],
                            "true": int(label),
                            "pred": int(pred_label),
                            "correct": int(label) == int(pred_label),
                        })
                seen += int(x.shape[0])
                if seen >= max_samples:
                    break
        if not class_energy:
            return {"skipped": "empty analysis energy"}
        centers_hz = (model.gabor.centers_cycles_per_sample().detach().cpu()[0] * float(model.fs_hz)).numpy()
        sigmas = model.gabor.sigmas().detach().cpu()[0].numpy()
        class_profiles = {str(k): torch.stack(v).mean(0).numpy() for k, v in sorted(class_energy.items())}
        profile_arr = np.stack(list(class_profiles.values()), axis=0)
        profile_norm = profile_arr / np.maximum(profile_arr.sum(axis=1, keepdims=True), 1e-8)
        subdir = build.paths.figure_subdir("gabor_interpretation")
        subdir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": model_name,
            "metrics": dict(final_metrics or {}),
            "center_frequency_hz": centers_hz.tolist(),
            "sigma_samples": sigmas.tolist(),
            "class_band_energy": {k: v.tolist() for k, v in class_profiles.items()},
            "sample_visualizations": [
                {"true": r["true"], "pred": r["pred"], "correct": r["correct"]}
                for r in sample_records
            ],
        }
        (subdir / f"learned_gabor_properties_{model_name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._plot_gabor_interpretation(subdir, model_name, centers_hz, sigmas, profile_norm, list(class_profiles.keys()))
        self._plot_signal_fault_evidence(subdir, model_name, centers_hz, sample_records[:6])
        return payload

    @staticmethod
    def _plot_gabor_interpretation(subdir: Path, model_name: str, centers_hz: np.ndarray, sigmas: np.ndarray, profile_norm: np.ndarray, label_names: Sequence[str]) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        for idx, label in enumerate(label_names):
            ax.plot(centers_hz, profile_norm[idx], marker="o", linewidth=1.2, markersize=2.5, label=f"class {label}")
        ax.set_xlabel("Learned Gabor center frequency (Hz)")
        ax.set_ylabel("Normalized envelope energy")
        ax.set_title(f"Class-wise Gabor envelope fingerprints: {model_name}")
        ax.grid(alpha=0.2)
        ax.legend(ncol=min(4, len(label_names)), fontsize=8)
        fig.tight_layout()
        fig.savefig(subdir / f"class_gabor_fingerprint_{model_name}.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        ax.scatter(centers_hz, sigmas, s=24)
        ax.set_xlabel("Learned Gabor center frequency (Hz)")
        ax.set_ylabel("Gabor sigma (samples)")
        ax.set_title(f"Q-parameterized Gabor geometry: {model_name}")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(subdir / f"gabor_geometry_{model_name}.png", dpi=300)
        plt.close(fig)

    @staticmethod
    def _plot_signal_fault_evidence(subdir: Path, model_name: str, centers_hz: np.ndarray, records: Sequence[Mapping[str, Any]]) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for idx, record in enumerate(records):
            x = record["x"].numpy()
            env = record["env"].numpy()
            band_energy = env.mean(axis=1)
            top_indices = np.argsort(band_energy)[-3:][::-1]
            title_state = "correct" if record["correct"] else "wrong"

            fig, axes = plt.subplots(3, 1, figsize=(9.0, 6.2), gridspec_kw={"height_ratios": [1.0, 1.0, 1.25]})
            t = np.arange(len(x))
            axes[0].plot(t, x, color="#2A6FBB", linewidth=0.8)
            axes[0].set_title(f"Input vibration signal | y={record['true']} pred={record['pred']} {title_state}")
            axes[0].set_ylabel("Amplitude")
            axes[0].grid(alpha=0.18)

            for band_idx in top_indices:
                curve = env[band_idx]
                curve = curve / max(float(curve.max()), 1e-8)
                axes[1].plot(t, curve, linewidth=1.0, label=f"{centers_hz[band_idx]:.1f} Hz")
            axes[1].set_title("Top Gabor envelope responses")
            axes[1].set_ylabel("Normalized envelope")
            axes[1].grid(alpha=0.18)
            axes[1].legend(fontsize=8, ncol=3)

            show_env = env / np.maximum(env.max(axis=1, keepdims=True), 1e-8)
            im = axes[2].imshow(show_env, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
            axes[2].set_title("Gabor envelope heatmap")
            axes[2].set_xlabel("Time index")
            axes[2].set_ylabel("Gabor band")
            fig.colorbar(im, ax=axes[2], fraction=0.025, pad=0.02)

            fig.tight_layout()
            suffix = "c" if record["correct"] else "w"
            fig.savefig(subdir / f"signal_fault_evidence_{model_name}_i{idx}_y{record['true']}_p{record['pred']}_{suffix}.png", dpi=300)
            plt.close(fig)


VibrationDiagnosisTrain = MotorBearingDiagnosisRunner


def train_diagnosis(dataset_name: str = "PU", train_subset: str = "ds1", test_subsets: Sequence[str] = ("ds1", "ds2", "ds3", "ds4"), cfg: Optional[TrainConfig] = None, selected_names: Sequence[str] | None = None) -> list[TrainerBuildResult]:
    return MotorBearingDiagnosisRunner(cfg=cfg).train_suite(dataset_name=dataset_name, train_subset=train_subset, test_subsets=test_subsets, selected_names=selected_names)


def train_loss_ablation(dataset_name: str = "PU", train_subset: str = "ds1", test_subsets: Sequence[str] = ("ds1", "ds2", "ds3", "ds4"), cfg: Optional[TrainConfig] = None, selected_names: Sequence[str] | None = None, sweep_values: Mapping[str, Sequence[float]] | None = None) -> list[TrainerBuildResult]:
    run_cfg = make_loss_ablation_config(cfg)
    return MotorBearingDiagnosisRunner(cfg=run_cfg).train_suite(specs=make_loss_ablation_specs(sweep_values=sweep_values), selected_names=selected_names, dataset_name=dataset_name, train_subset=train_subset, test_subsets=test_subsets)


def train_structure_ablation(dataset_name: str = "PU", train_subset: str = "ds1",
                             test_subsets: Sequence[str] = ("ds1", "ds2", "ds3","ds4"),  #
                             cfg: Optional[TrainConfig] = None,
                             selected_names: Sequence[str] | None = None, *,
                             lambda_decor: float = 1e-3,
                             lambda_impulse: float = 1e-4,
                             label_smoothing: float = 0.05) -> list[TrainerBuildResult]:
    run_cfg = make_structure_ablation_config(cfg)
    specs = make_structure_ablation_specs(lambda_decor=lambda_decor, lambda_impulse=lambda_impulse, label_smoothing=label_smoothing)
    return MotorBearingDiagnosisRunner(cfg=run_cfg).train_suite(specs=specs, selected_names=selected_names, dataset_name=dataset_name, train_subset=train_subset, test_subsets=test_subsets)


def train_single_ablation(name: str = "msgabor_full", dataset_name: str = "PU",
                          train_subset: str = "ds1",
                          test_subsets: Sequence[str] = ("ds1", "ds2", "ds3", "ds4"),
                          cfg: Optional[TrainConfig] = None) -> TrainerBuildResult:
    specs = {spec.name: spec for spec in make_sci_ablation_specs()}
    if name not in specs:
        raise KeyError(f"Unknown ablation name: {name}. Available={sorted(specs)}")
    return MotorBearingDiagnosisRunner(cfg=cfg).train_one(spec=specs[name], dataset_name=dataset_name, train_subset=train_subset, test_subsets=test_subsets)


def run() -> list[TrainerBuildResult]:
    # return train_structure_ablation()
    train_structure_ablation(); SEED.pop(0)
    train_structure_ablation(); SEED.pop(0)
    train_structure_ablation(); SEED.pop(0)
    train_structure_ablation(); SEED.pop(0)
    # return train_loss_ablation()



if __name__ == "__main__":
    run()
