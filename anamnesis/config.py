"""Configuration for Llama 3.1 8B Instruct extraction.

Key parameters:
  - Model: Llama 3.1 8B Instruct (32 layers, 4096 hidden, 32 query heads)
  - Temperature: 0.6 (model's native)
  - EOS tokens: [128001, 128008, 128009] (includes <|eom_id|>)
  - Sampled layers: [0, 8, 16, 20, 24, 28, 31] (proportional depth sampling)
  - dtype: bfloat16 (model's native)
  - 200 samples: 20 topics × 5 modes × 2 reps
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from anamnesis.modes.run4_modes import (
    RUN4_MODE_INDEX as MODE_INDEX,
    RUN4_MODES as PROCESSING_MODES,
)


# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent

# Legacy data root — for accessing 3B experiment data from earlier runs.
# Set ANAMNESIS_LEGACY_DATA to override (e.g., path to old phase_0 outputs).
LEGACY_DATA_ROOT = Path(os.environ.get(
    "ANAMNESIS_LEGACY_DATA",
    str(PROJECT_ROOT.parent / "phase_0"),
))

# Run versioning
RUN_NAME: str = os.environ.get("ANAMNESIS_RUN_NAME", "run_8b_baseline")

OUTPUTS_BASE = PROJECT_ROOT / "outputs"
CALIBRATION_DIR = OUTPUTS_BASE / "calibration" / "llama31_8b"
OUTPUTS_DIR = OUTPUTS_BASE / "runs" / RUN_NAME
SIGNATURES_DIR = OUTPUTS_DIR / "signatures"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PROMPTS_PATH = PROJECT_ROOT / "prompts" / "prompt_sets.json"


# ── Model ──────────────────────────────────────────────────────────────────────

class ModelConfig(BaseModel):
    """Configuration for Llama 3.1 8B Instruct."""

    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    torch_dtype: str = "bfloat16"  # model's native dtype
    attn_implementation: str = "eager"  # required — flash/sdpa don't return attn weights
    device_map: str = "auto"

    # Architecture constants (Llama 3.1 8B Instruct)
    num_layers: int = 32
    hidden_dim: int = 4096
    num_attention_heads: int = 32
    num_kv_heads: int = 8  # GQA — same ratio as 3B (4:1)
    head_dim: int = 128
    vocab_size: int = 128256


# ── Generation ─────────────────────────────────────────────────────────────────

class GenerationConfig(BaseModel):
    """Parameters for model.generate()."""

    max_new_tokens: int = 512
    temperature: float = 0.6  # model's native (3B used 0.7)
    top_p: float = 0.9
    do_sample: bool = True

    # Llama 3.1 8B Instruct stop tokens:
    #   128001 = <|end_of_text|>
    #   128008 = <|eom_id|> (end of message, new in 3.1)
    #   128009 = <|eot_id|> (end of turn)
    eos_token_ids: list[int] = Field(
        default=[128001, 128008, 128009],
        description="Token IDs that signal end of generation",
    )

    # These MUST be True for extraction
    output_hidden_states: bool = True
    output_attentions: bool = True
    output_logits: bool = True
    return_dict_in_generate: bool = True


# ── Extraction ─────────────────────────────────────────────────────────────────

class ExtractionConfig(BaseModel):
    """Controls which features to extract and how.

    Sampled layers for 8B (32 layers) follow the same proportional
    sampling as 3B (28 layers): [0%, 25%, 50%, 63%, 75%, 88%, 97%],
    denser at 60-80% depth per the Pochinkov finding.

    3B: [0, 7, 14, 18, 21, 24, 27]  for 28 layers
    8B: [0, 8, 16, 20, 24, 28, 31]  for 32 layers
    """

    sampled_layers: list[int] = Field(
        default=[0, 8, 16, 20, 24, 28, 31],
        description="Layers to extract KV cache and spectral features from",
    )

    # Tier 3 PCA layers — proportional to 3B's [7, 14, 18, 21, 24]
    pca_layers: list[int] = Field(
        default=[8, 16, 20, 24, 28],
        description="Layers for residual stream PCA projection",
    )
    pca_components: int = 50
    pca_temporal_samples: int = 5

    # Temporal sampling for trajectories
    trajectory_points: int = 5

    # Spectral features
    spectral_subsample_step: int = 10

    # KV cache epoch detection
    epoch_window_size: int = 50
    epoch_stride: int = 25

    # Bayesian surprise
    surprise_window: int = 20
    surprise_threshold_sigma: float = 1.5

    # kNN-LM baseline
    knnlm_pca_components: int = 100

    # Enable/disable tiers
    enable_tier1: bool = True
    enable_tier2: bool = True
    enable_tier2_5: bool = True
    enable_tier3: bool = True
    enable_knnlm_baseline: bool = True

    # Cross-layer agreement thresholds (proportional to model depth)
    # These replace the hardcoded l<=7 / l>=21 from the 3B model
    early_layer_cutoff: int = Field(
        default=8,
        description="Layers <= this are 'early' for cross-layer agreement (first quarter)",
    )
    late_layer_cutoff: int = Field(
        default=24,
        description="Layers >= this are 'late' for cross-layer agreement (last quarter)",
    )

    # Raw tensor saving (for GPU-free feature iteration)
    save_raw_tensors: bool = Field(
        default=False,
        description="Save raw per-token tensors alongside feature vectors",
    )
    raw_logits_top_k: int = Field(
        default=50,
        description="Number of top logits to save per timestep (saves 99.96% space)",
    )
    raw_hidden_dtype: str = Field(
        default="float16",
        description="Dtype for saved hidden states and attention (float16 halves disk)",
    )


# ── Feature Pipeline (v2) ──────────────────────────────────────────────────────

class FeaturePipelineConfig(BaseModel):
    """Configuration for pluggable feature families (v2 pipeline).

    Controls which feature families are enabled when computing features
    from saved raw tensors. Each family adds features on top of the
    baseline T1/T2/T2.5/T3 tiers.
    """

    # Whether to include baseline tiers (T1/T2/T2.5/T3 from state_extractor)
    include_baseline_tiers: bool = Field(
        default=True,
        description="Include baseline T1/T2/T2.5/T3 features",
    )

    # Residual stream trajectory features
    enable_residual_trajectory: bool = Field(
        default=False,
        description="Extract trajectory features (velocity, curvature, directness)",
    )
    trajectory_layers: list[int] = Field(
        default=[8, 16, 20, 24, 28],
        description="Layers for trajectory feature computation",
    )

    # Contrastive projection
    enable_contrastive_projection: bool = Field(
        default=False,
        description="Apply trained contrastive projection to hidden states",
    )
    contrastive_model_path: Path | None = Field(
        default=None,
        description="Path to trained contrastive projection model (.pt)",
    )
    contrastive_layers: list[int] = Field(
        default=[8, 16, 20, 24, 28],
        description="Layers for contrastive projection",
    )
    contrastive_temporal_samples: int = Field(
        default=5,
        description="Number of temporal samples for contrastive projection",
    )

    # Attention flow features
    enable_attention_flow: bool = Field(
        default=False,
        description="Extract attention flow features (region decomp, head diversity)",
    )

    # Gate features (SwiGLU)
    enable_gate_features: bool = Field(
        default=False,
        description="Extract SwiGLU gate activation features",
    )
    gate_sparsity_threshold: float = Field(
        default=0.01,
        description="Threshold for gate activation to count as 'active'",
    )

    # Temporal dynamics (windowed T2/T2.5 metrics)
    enable_temporal_dynamics: bool = Field(
        default=False,
        description="Extract windowed temporal decomposition of core T2/T2.5 metrics",
    )

    # Per-head heterogeneity (preserve what head-averaging destroys)
    enable_per_head: bool = Field(
        default=False,
        description="Extract per-head attention/key heterogeneity features",
    )

    # v_proj value-vector geometry (OV-circuit storage surface; never featurized before)
    enable_value_geometry: bool = Field(
        default=False,
        description="Extract v_proj value-vector geometry (spread/eff_dim/drift/novelty + basis-free stats)",
    )

    # Query / QK-space content geometry (pre-RoPE query trajectory + q·k content alignment)
    enable_qk_geometry: bool = Field(
        default=False,
        description="Extract pre-RoPE query geometry + q·k content alignment (RoPE-invariant self-align)",
    )

    # Cross-layer KV-cache CKA (basis-invariant key/value structure agreement across depth; replaces C4)
    enable_kv_cka: bool = Field(
        default=False,
        description="Extract cross-layer linear CKA for keys and values (basis-invariant)",
    )

    # AttnRes routing (kotodama Block Attention Residuals; kotodama-only — needs attn_res_* capture fields)
    enable_attn_res: bool = Field(
        default=False,
        description="Extract AttnRes cross-block routing features (anchor-vs-recency, concentration, committed geom)",
    )

    # MoE expert routing (vmb arm A7, M6 DeepSeek-V2-Lite class — needs router_dist/router_branch_norms
    # capture fields; dense models leave them None and this family returns empty)
    enable_expert_routing: bool = Field(
        default=False,
        description="Extract MoE expert-routing features (alloc entropy/margin/coverage/load, shared-mass, switch/drift)",
    )
    expert_routing_top_k: int = Field(
        default=6,
        description="num_experts_per_tok — selection size for coverage/load/drift (6 for DeepSeek-V2-Lite)",
    )

    # Temporal operator settings (shared across families)
    temporal_n_windows: int = Field(
        default=4,
        description="Number of temporal windows for windowed stats",
    )
    enable_stft: bool = Field(
        default=True,
        description="Include STFT spectral features in temporal operators",
    )
    stft_nperseg: int = Field(
        default=64,
        description="STFT window length (reduced for short generations)",
    )


# ── Calibration ────────────────────────────────────────────────────────────────

class CalibrationConfig(BaseModel):
    """Settings for positional decomposition calibration."""

    num_calibration_prompts: int = 50
    calibration_max_tokens: int = 512
    positional_means_path: Path = Field(
        default=CALIBRATION_DIR / "positional_means.npz",
    )
    pca_model_path: Path = Field(
        default=CALIBRATION_DIR / "pca_model.pkl",
    )


# ── Model presets ──────────────────────────────────────────────────────────────

class ModelPreset(BaseModel):
    """Per-model architecture + sampling defaults.

    Single source of truth for everything that varies between Llama 3.1 8B
    and Llama 3.2 3B (architecture, sampled layers, decode params,
    calibration paths). Consumed by `scripts.run_extraction` and
    `extraction.feature_pipeline` so layer presets stay in sync across
    extraction and feature recomputation.
    """

    model_id: str
    torch_dtype: str
    num_layers: int
    hidden_dim: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    sampled_layers: list[int]
    pca_layers: list[int]
    trajectory_layers: list[int]
    contrastive_layers: list[int]
    early_layer_cutoff: int
    late_layer_cutoff: int
    temperature: float
    eos_token_ids: list[int]
    calibration_dir: Path
    # Per-sampled-layer attention type for interleaved-attention architectures
    # (Gemma-3 class: 5 local sliding-window : 1 global). None = all-global
    # (Llama/Qwen/OLMo full-context attention at every layer). PRE-COMMITTED
    # (outer-agent ruling 2026-07-12, before M5 floors): local-window layers are
    # structurally recency-dominated, so CROSS-MODEL attention-source
    # comparisons — the source-ordering L-rung test in particular — use GLOBAL
    # layers primarily; local-layer attention cells are per-model exploratory.
    attention_layer_types: dict[int, str] | None = None


MODEL_PRESETS: dict[str, ModelPreset] = {
    "8b": ModelPreset(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype="bfloat16",
        num_layers=32,
        hidden_dim=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        sampled_layers=[0, 8, 16, 20, 24, 28, 31],
        pca_layers=[8, 16, 20, 24, 28],
        trajectory_layers=[8, 16, 20, 24, 28],
        contrastive_layers=[8, 16, 20, 24, 28],
        early_layer_cutoff=8,
        late_layer_cutoff=24,
        temperature=0.6,
        eos_token_ids=[128001, 128008, 128009],
        calibration_dir=OUTPUTS_BASE / "calibration" / "llama31_8b",
    ),
    "3b": ModelPreset(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype="float16",
        num_layers=28,
        hidden_dim=3072,
        num_attention_heads=24,
        num_kv_heads=8,
        head_dim=128,
        sampled_layers=[0, 7, 14, 18, 21, 24, 27],
        pca_layers=[7, 14, 18, 21, 24],
        trajectory_layers=[7, 14, 18, 21, 24],
        contrastive_layers=[7, 14, 18, 21, 24],
        early_layer_cutoff=7,
        late_layer_cutoff=21,
        temperature=0.7,
        eos_token_ids=[128001, 128009],
        calibration_dir=LEGACY_DATA_ROOT / "outputs" / "calibration",
    ),
    "olmo2-7b": ModelPreset(
        # vmb M4 (prereg §2c model roster): RLHF-free machinery isolation.
        # BASE model — no chat template (bare prompts only; run_gen_tokens
        # refuses system prompts for template-less models). Full MHA
        # (num_kv_heads == num_heads): the GQA indexing caveat does not apply.
        # OLMo-2 applies q/k RMSNorm between the projections and RoPE, so
        # k_proj hooks capture PRE-NORM pre-RoPE keys (position-free holds;
        # substrate differs from Llama's post-proj keys — onboarding audit
        # 2026-07-12, journal wave1-continuation W4).
        model_id="allenai/OLMo-2-1124-7B",
        torch_dtype="bfloat16",
        num_layers=32,
        hidden_dim=4096,
        num_attention_heads=32,
        num_kv_heads=32,
        head_dim=128,
        sampled_layers=[0, 8, 16, 20, 24, 28, 31],
        pca_layers=[8, 16, 20, 24, 28],
        trajectory_layers=[8, 16, 20, 24, 28],
        contrastive_layers=[8, 16, 20, 24, 28],
        early_layer_cutoff=8,
        late_layer_cutoff=24,
        temperature=0.7,
        eos_token_ids=[100257],
        calibration_dir=OUTPUTS_BASE / "calibration" / "olmo2_7b",
    ),
    "gemma3-27b": ModelPreset(
        # vmb M5 (prereg roster): scale rung + THIRD architecture family.
        # Gemma3ForConditionalGeneration (multimodal wrapper) — decoder layers
        # resolve via extraction.model_loader.decoder_layers(); GPU validation
        # of the load path PENDING (staged 2026-07-12 while the GPU box is lent out).
        # 5:1 local:global attention interleave (sliding_window=1024; global at
        # (i+1) % 6 == 0): sampled layers prefer GLOBAL (only layer 0 is local,
        # kept for the early-band anchor point). Native sampling per the model
        # card: temperature 1.0 (Stage-0 chain should pass --override-top-p
        # 0.95 to match; pipeline default is 0.9).
        model_id="google/gemma-3-27b-it",
        torch_dtype="bfloat16",
        num_layers=62,
        hidden_dim=5376,
        num_attention_heads=32,
        num_kv_heads=16,
        head_dim=128,
        sampled_layers=[0, 11, 23, 35, 41, 53, 59],
        pca_layers=[11, 23, 35, 41, 53],
        trajectory_layers=[11, 23, 35, 41, 53],
        contrastive_layers=[11, 23, 35, 41, 53],
        early_layer_cutoff=15,
        late_layer_cutoff=46,
        temperature=1.0,
        eos_token_ids=[1, 106],
        calibration_dir=OUTPUTS_BASE / "calibration" / "gemma3_27b",
        attention_layer_types={0: "local", 11: "global", 23: "global",
                               35: "global", 41: "global", 53: "global",
                               59: "global"},
    ),
    "qwen-7b": ModelPreset(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        torch_dtype="bfloat16",
        num_layers=28,
        hidden_dim=3584,
        num_attention_heads=28,
        num_kv_heads=4,
        head_dim=128,
        sampled_layers=[0, 7, 14, 18, 21, 24, 27],
        pca_layers=[7, 14, 18, 21, 24],
        trajectory_layers=[7, 14, 18, 21, 24],
        contrastive_layers=[7, 14, 18, 21, 24],
        early_layer_cutoff=7,
        late_layer_cutoff=21,
        temperature=0.7,
        eos_token_ids=[151643, 151645],
        calibration_dir=OUTPUTS_BASE / "calibration" / "qwen25_7b",
    ),
    "dsv2-lite": ModelPreset(
        # vmb M6 (prereg roster + ADDENDUM 2026-07-17t pin): DeepSeek-V2-Lite-Chat —
        # FIRST MoE model (2 shared + 64 routed experts, top-6 GREEDY; first_k_dense_replace=1,
        # so layer 0 is a dense MLP and layers 1–26 are MoE). Loaded via the NATIVE transformers
        # `deepseek_v2` integration (model_type=deepseek_v2, transformers 5.3) — load with
        # trust_remote_code=False; the bundled auto_map remote code (modeling_deepseek.py) has a
        # DIFFERENT internal structure (per-expert gate_proj) that the M6 hooks do NOT target.
        # MLA attention: keys-source analog = the c_KV compressed latent (kv_a_proj_with_mqa[:512],
        # position-free); k_pe (64) banked, not featurized. No k_proj/v_proj module and q_proj is
        # 192-dim/head (qk_nope 128 + qk_rope 64) — queries/values are NOT wave-1 features for M6.
        # All numbers below VERIFIED against the downloaded config.json (2026-07-17, on-node).
        model_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
        torch_dtype="bfloat16",
        num_layers=27,
        hidden_dim=2048,
        num_attention_heads=16,
        num_kv_heads=16,           # config num_key_value_heads; MLA latent (attention weights are 16-head).
                                   # NB the keys-SOURCE capture is the 512-d c_KV latent, not these 16 heads.
        head_dim=128,              # v_head_dim (config has no top-level head_dim). qk_head_dim=192.
        sampled_layers=[0, 5, 11, 15, 18, 22, 26],   # proportional-depth, mid-heavy; L0 dense → 0 xrt features
        pca_layers=[5, 11, 15, 18, 22],
        trajectory_layers=[5, 11, 15, 18, 22],
        contrastive_layers=[5, 11, 15, 18, 22],
        early_layer_cutoff=7, late_layer_cutoff=20,
        temperature=0.3,           # CORRECTED from spec §4's expected 0.7 — generation_config.json = 0.3
                                   # (native top_p=0.95: Stage-0 chain passes --override-top-p 0.95, à la Gemma)
        eos_token_ids=[100001],    # <｜end▁of▁sentence｜>; single EOS, no separate end-of-turn token
        calibration_dir=OUTPUTS_BASE / "calibration" / "dsv2_lite",
    ),
}


# ── Run registry ───────────────────────────────────────────────────────────────

class RunSpec(BaseModel):
    """A named extraction run that the unified analysis can target.

    Centralises the canonical signature paths (and any addon directories
    holding split-out feature families) so callers don't have to hardcode
    layout in scripts.
    """

    name: str
    signature_dir: Path
    addon_dirs: list[Path] = Field(default_factory=list)
    description: str = ""


RUNS: dict[str, RunSpec] = {
    "8b_baseline": RunSpec(
        name="8b_baseline",
        signature_dir=Path("outputs/runs/run_8b_baseline/signatures"),
        description="Baseline 8B extraction (5 format-controlled modes).",
    ),
    "3b_run4": RunSpec(
        name="3b_run4",
        signature_dir=LEGACY_DATA_ROOT / "outputs" / "runs"
            / "run4_format_controlled" / "signatures",
        description="Phase-0 3B run4 (format-controlled).",
    ),
    "8b_v2": RunSpec(
        name="8b_v2",
        signature_dir=Path("outputs/runs/8b_fat_01/signatures_v2"),
        addon_dirs=[
            Path("outputs/runs/8b_fat_01/signatures_v2_addon"),
            Path("outputs/runs/8b_fat_01/signatures_v2_contrastive"),
        ],
        description="8B v2 feature-pipeline with engineered families + contrastive.",
    ),
    "3b_v2": RunSpec(
        name="3b_v2",
        signature_dir=Path("outputs/runs/3b_fat_01/signatures_v2"),
        addon_dirs=[
            Path("outputs/runs/3b_fat_01/signatures_v2_contrastive"),
        ],
        description="3B v2 feature-pipeline.",
    ),
}


# ── Experiment ─────────────────────────────────────────────────────────────────

ProcessingMode = Literal[
    "linear",
    "analogical",
    "socratic",
    "contrastive",
    "dialectical",
]

# Canonical mode prompts and index live in `anamnesis.modes.run4_modes` and are
# re-exported at the top of this module so consumers that already import them
# from `anamnesis.config` keep working.


class GenerationSpec(BaseModel):
    """Full specification for a single generation run.

    mode is str (not ProcessingMode literal) to support extended modes
    (structured, compressed, associative) and prompt-swap modes
    (swap_socratic→linear, etc.) from the unified extraction script.
    """

    generation_id: int
    prompt_set: str
    topic: str
    topic_idx: int
    mode: str
    mode_idx: int
    system_prompt: str
    user_prompt: str
    seed: int
    repetition: int = 0


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)

    outputs_dir: Path = OUTPUTS_DIR
    signatures_dir: Path = SIGNATURES_DIR
    figures_dir: Path = FIGURES_DIR
    prompts_path: Path = PROMPTS_PATH
    metadata_path: Path = Field(default=OUTPUTS_DIR / "metadata.json")
    results_path: Path = Field(default=OUTPUTS_DIR / "results.json")

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.signatures_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
