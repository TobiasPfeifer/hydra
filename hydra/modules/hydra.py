# Copyright (c) 2024, Sukjun Hwang, Aakash Lahoti, Ratish Puduppully, Tri Dao, Albert Gu.
# Base code from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2_simple.py

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from typing import Dict, Optional, Set, Tuple

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
except ImportError:
    RMSNormGated = None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from .ops import hydra_split_conv1d_scan_combined


_TRACE_ENABLED = os.environ.get("ESM_TRACE_BISECT", "0") == "1"
_TRACE_FAIL_FAST = os.environ.get("ESM_TRACE_FAIL_FAST", "1") == "1"
_TRACE_MAX_STEPS = int(os.environ.get("ESM_TRACE_MAX_STEPS", "1"))
_TRACE_RANK0_ONLY = os.environ.get("ESM_TRACE_RANK0_ONLY", "1") == "1"
_TRACE_LOG_FORWARD = os.environ.get("ESM_TRACE_LOG_FORWARD", "1") == "1"
_TRACE_LOG_BACKWARD = os.environ.get("ESM_TRACE_LOG_BACKWARD", "1") == "1"
_TRACE_LAYERS_RAW = os.environ.get("ESM_TRACE_LAYERS", "all")
_TRACE_POINTS_RAW = os.environ.get("ESM_TRACE_POINTS", "all")


def _parse_int_filter(raw_value: str) -> Optional[Set[int]]:
    value = raw_value.strip().lower()
    if value in {"", "all", "*"}:
        return None
    values: Set[int] = set()
    for token in raw_value.split(","):
        token = token.strip()
        if token:
            values.add(int(token))
    return values


def _parse_str_filter(raw_value: str) -> Optional[Set[str]]:
    value = raw_value.strip().lower()
    if value in {"", "all", "*"}:
        return None
    values: Set[str] = set()
    for token in raw_value.split(","):
        token = token.strip()
        if token:
            values.add(token)
    return values


_TRACE_LAYERS = _parse_int_filter(_TRACE_LAYERS_RAW)
_TRACE_POINTS = _parse_str_filter(_TRACE_POINTS_RAW)

_TRACE_STATE: Dict[str, object] = {
    "forward_step": 0,
    "active_step": 0,
    "grad_norms": {},      # (step, layer, point) -> norm
    "parents": {},         # key -> parent_key
    "children": {},        # parent_key -> set(child_key)
    "logged_gains": set(), # (child_key, parent_key)
}


def _is_rank0() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def _trace_rank_enabled() -> bool:
    if not _TRACE_ENABLED:
        return False
    if not _TRACE_RANK0_ONLY:
        return True
    return _is_rank0()


def _trace_layer_enabled(layer_idx: Optional[int]) -> bool:
    if not _trace_rank_enabled():
        return False
    if layer_idx is None:
        return False
    if _TRACE_LAYERS is None:
        return True
    return layer_idx in _TRACE_LAYERS


def _trace_point_enabled(point: str) -> bool:
    if _TRACE_POINTS is None:
        return True
    return point in _TRACE_POINTS


def _tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        flat = x.detach().float().reshape(-1)
        if flat.numel() == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "abs_max": 0.0,
                "norm": 0.0,
                "finite_ratio": 1.0,
            }
        finite = torch.isfinite(flat)
        finite_ratio = float(finite.float().mean().item())
        if finite.any():
            finite_flat = flat[finite]
            mean = float(finite_flat.mean().item())
            std = float(finite_flat.std(unbiased=False).item())
            abs_max = float(finite_flat.abs().max().item())
            norm = float(finite_flat.norm().item())
        else:
            mean = float("nan")
            std = float("nan")
            abs_max = float("nan")
            norm = float("nan")
        return {
            "mean": mean,
            "std": std,
            "abs_max": abs_max,
            "norm": norm,
            "finite_ratio": finite_ratio,
        }


def _edge_gain(child_norm: float, parent_norm: float) -> float:
    eps = 1e-30
    return parent_norm / (child_norm + eps)


def _format_key(key: Tuple[int, int, str]) -> str:
    _, layer_idx, point = key
    return f"layer={layer_idx} point={point}"


def _log_gain_if_ready(child_key: Tuple[int, int, str], parent_key: Tuple[int, int, str]) -> None:
    grad_norms: Dict[Tuple[int, int, str], float] = _TRACE_STATE["grad_norms"]  # type: ignore[assignment]
    logged_gains: Set[Tuple[Tuple[int, int, str], Tuple[int, int, str]]] = _TRACE_STATE["logged_gains"]  # type: ignore[assignment]
    if child_key not in grad_norms or parent_key not in grad_norms:
        return
    pair = (child_key, parent_key)
    if pair in logged_gains:
        return
    logged_gains.add(pair)
    child_norm = grad_norms[child_key]
    parent_norm = grad_norms[parent_key]
    gain = _edge_gain(child_norm, parent_norm)
    step = child_key[0]
    print(
        f"[TRACE_GAIN] step={step:03d} edge={child_key[2]}->{parent_key[2]} "
        f"child_norm={child_norm:.4e} parent_norm={parent_norm:.4e} gain={gain:.4e}",
        flush=True,
    )


class Hydra(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=7,
        conv_init=None,
        expand=2,
        headdim=64,
        ngroups=1,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=False,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * (2 * self.ngroups * self.d_state) + 2 * self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * (2 * self.ngroups * self.d_state)
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv // 2,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        A = torch.ones(self.nheads, dtype=torch.float32, device=device)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True
        self.fc_D = nn.Linear(self.d_inner, self.nheads, bias=False, **factory_kwargs)

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=True, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def _trace_context(self) -> tuple[bool, int]:
        if not _trace_rank_enabled():
            return False, 0
        if self.layer_idx == 1:
            _TRACE_STATE["forward_step"] = int(_TRACE_STATE["forward_step"]) + 1
            _TRACE_STATE["active_step"] = int(_TRACE_STATE["forward_step"])
            _TRACE_STATE["grad_norms"] = {}
            _TRACE_STATE["parents"] = {}
            _TRACE_STATE["children"] = {}
            _TRACE_STATE["logged_gains"] = set()
            active_step = int(_TRACE_STATE["active_step"])
            if active_step <= _TRACE_MAX_STEPS:
                print(f"[TRACE] ===== begin step {active_step} =====", flush=True)
        active_step = int(_TRACE_STATE["active_step"])
        if active_step == 0 or active_step > _TRACE_MAX_STEPS:
            return False, active_step
        if not _trace_layer_enabled(self.layer_idx):
            return False, active_step
        return True, active_step

    def _trace_tensor(
        self,
        point: str,
        tensor: torch.Tensor,
        enabled: bool,
        step: int,
        parent: Optional[str] = None,
    ) -> torch.Tensor:
        if not enabled or not _trace_point_enabled(point):
            return tensor

        layer_idx = int(self.layer_idx) if self.layer_idx is not None else -1
        key = (step, layer_idx, point)

        if parent is not None:
            parent_key = (step, layer_idx, parent)
            parents: Dict[Tuple[int, int, str], Tuple[int, int, str]] = _TRACE_STATE["parents"]  # type: ignore[assignment]
            children: Dict[Tuple[int, int, str], Set[Tuple[int, int, str]]] = _TRACE_STATE["children"]  # type: ignore[assignment]
            parents[key] = parent_key
            children.setdefault(parent_key, set()).add(key)

        stats = _tensor_stats(tensor)
        if _TRACE_LOG_FORWARD:
            print(
                f"[TRACE_FWD] step={step:03d} layer={layer_idx:02d} point={point:24s} "
                f"mean={stats['mean']:+.4e} std={stats['std']:.4e} abs_max={stats['abs_max']:.4e} "
                f"norm={stats['norm']:.4e} finite_ratio={stats['finite_ratio']:.6f}",
                flush=True,
            )
        if _TRACE_FAIL_FAST and stats["finite_ratio"] < 1.0:
            raise RuntimeError(
                f"Non-finite forward tensor at step={step}, layer={layer_idx}, point={point}"
            )

        if tensor.requires_grad:
            def _grad_hook(grad: torch.Tensor) -> torch.Tensor:
                grad_stats = _tensor_stats(grad)
                grad_norms: Dict[Tuple[int, int, str], float] = _TRACE_STATE["grad_norms"]  # type: ignore[assignment]
                grad_norms[key] = grad_stats["norm"]
                if _TRACE_LOG_BACKWARD:
                    print(
                        f"[TRACE_BWD] step={step:03d} layer={layer_idx:02d} point={point:24s} "
                        f"mean={grad_stats['mean']:+.4e} std={grad_stats['std']:.4e} abs_max={grad_stats['abs_max']:.4e} "
                        f"norm={grad_stats['norm']:.4e} finite_ratio={grad_stats['finite_ratio']:.6f}",
                        flush=True,
                    )
                if _TRACE_FAIL_FAST and grad_stats["finite_ratio"] < 1.0:
                    raise RuntimeError(
                        f"Non-finite backward tensor at step={step}, layer={layer_idx}, point={point}"
                    )

                parents: Dict[Tuple[int, int, str], Tuple[int, int, str]] = _TRACE_STATE["parents"]  # type: ignore[assignment]
                children: Dict[Tuple[int, int, str], Set[Tuple[int, int, str]]] = _TRACE_STATE["children"]  # type: ignore[assignment]
                if key in parents:
                    _log_gain_if_ready(key, parents[key])
                for child_key in children.get(key, set()):
                    _log_gain_if_ready(child_key, key)
                return grad

            tensor.register_hook(_grad_hook)
        return tensor

    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape
        trace_enabled, trace_step = self._trace_context()
        u = self._trace_tensor("u_in", u, trace_enabled, trace_step)

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        zxbcdt = self._trace_tensor("zxbcdt", zxbcdt, trace_enabled, trace_step, parent="u_in")
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        A = self._trace_tensor("A", A, trace_enabled, trace_step)
        initial_states = repeat(self.init_states, "... -> b ...", b=2*batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path:
            out = hydra_split_conv1d_scan_combined(
                zxbcdt,
                self.conv1d.weight,
                self.conv1d.bias,
                self.dt_limit,
                self.dt_bias,
                A,
                self.fc_D.weight,
                self.D,
                self.norm.weight,
                self.norm.eps,
                self.out_proj.weight,
                self.out_proj.bias,
                self.chunk_size,
                initial_states,
                seq_idx,
                self.d_inner,
                self.d_state,
                self.headdim,
                self.ngroups,
            )
            out = self._trace_tensor("out_mem_eff", out, trace_enabled, trace_step, parent="zxbcdt")
            return out

        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * (2 * self.ngroups * self.d_state), 2 * self.nheads],
            dim=-1
        )
        z = self._trace_tensor("z", z, trace_enabled, trace_step, parent="zxbcdt")
        xBC = self._trace_tensor("xBC_preconv", xBC, trace_enabled, trace_step, parent="zxbcdt")
        dt = self._trace_tensor("dt_raw", dt, trace_enabled, trace_step, parent="zxbcdt")

        dt = torch.cat((dt[:, :, :self.nheads], torch.flip(dt[:, :, self.nheads:], (1,))), dim=0)
        dt = self._trace_tensor("dt_cat", dt, trace_enabled, trace_step, parent="dt_raw")
        dt = F.softplus(dt + self.dt_bias)  # (2 * B, L, nheads)
        dt = self._trace_tensor("dt_softplus", dt, trace_enabled, trace_step, parent="dt_cat")
        assert self.activation in ["silu", "swish"]

        # 1D Convolution
        xBC_conv = self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
        xBC_conv = self._trace_tensor("xBC_conv", xBC_conv, trace_enabled, trace_step, parent="xBC_preconv")
        xBC = self.act(xBC_conv)  # (B, L, self.d_inner + 2 * (2 * ngroups * d_state))
        xBC = self._trace_tensor("xBC_postact", xBC, trace_enabled, trace_step, parent="xBC_conv")

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, BC = torch.split(xBC, [self.d_inner, 2 * (2 * self.ngroups * self.d_state)], dim=-1)
        x_og = x
        x_og = self._trace_tensor("x_og", x_og, trace_enabled, trace_step, parent="xBC_postact")
        x = torch.cat((x, torch.flip(x, (1,))), dim=0)
        x = self._trace_tensor("x_stacked", x, trace_enabled, trace_step, parent="x_og")
        BC = torch.cat(
            (BC[:, :, :2 * self.ngroups * self.d_state],
             torch.flip(BC[:, :, 2 * self.ngroups * self.d_state:], (1,))),
            dim=0
        )
        BC = self._trace_tensor("BC_stacked", BC, trace_enabled, trace_step, parent="xBC_postact")
        B, C = torch.split(BC, [self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        B = self._trace_tensor("B", B, trace_enabled, trace_step, parent="BC_stacked")
        C = self._trace_tensor("C", C, trace_enabled, trace_step, parent="BC_stacked")

        scan = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=None,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
            **dt_limit_kwargs,
        )
        scan = self._trace_tensor("scan_raw", scan, trace_enabled, trace_step, parent="x_stacked")
        y = rearrange(scan, "b l h p -> b l (h p)")
        y = self._trace_tensor("scan_flat", y, trace_enabled, trace_step, parent="scan_raw")
        y = torch.roll(y, shifts=1, dims=1)
        y[:, 0, :] = 0.0
        y = self._trace_tensor("scan_shifted", y, trace_enabled, trace_step, parent="scan_flat")
        y_fw, y_bw = y[:batch], torch.flip(y[batch:], (1,))
        y_bidir = y_fw + y_bw
        y_bidir = self._trace_tensor("y_bidir", y_bidir, trace_enabled, trace_step, parent="scan_shifted")
        d_skip = x_og * repeat(
            F.linear(x_og, self.fc_D.weight, bias=self.D), "b l h -> b l (h p)", p=self.headdim
        )
        d_skip = self._trace_tensor("d_skip", d_skip, trace_enabled, trace_step, parent="x_og")

        # Gated norm on scan output only (d_skip added after to avoid
        # quadratic outliers inflating the normalized values and amplifying
        # the z-gate backward gradient)
        y = self.norm(y_bidir, z)
        y = self._trace_tensor("y_post_norm", y, trace_enabled, trace_step, parent="y_bidir")
        y = y + d_skip
        y = self._trace_tensor("y_pre_proj", y, trace_enabled, trace_step, parent="y_post_norm")
        out = self.out_proj(y)
        out = self._trace_tensor("out", out, trace_enabled, trace_step, parent="y_pre_proj")

        return out
