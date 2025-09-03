import json, pickle, math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Literal, Any, Optional

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer, QuantileTransformer
)
from sklearn.impute import SimpleImputer


class _CatLogits:
    """Utility to convert one-hot ⇄ logits per *single* feature."""
    EPS = 1e-4
    @staticmethod
    def fwd(oh: np.ndarray) -> np.ndarray:          # one-hot → logits
        return logit(np.clip((oh + _CatLogits.EPS) /
                             (1 + _CatLogits.EPS),
                             _CatLogits.EPS, 1 - _CatLogits.EPS))

    @staticmethod
    def inv(logits: np.ndarray) -> np.ndarray:
        """
        logits → one-hot (Gumbel-Softmax)
        During training you want a *high* τ (≈ 1.0) for smoother gradients,
        and during pure sampling you want τ→0 for sharp argmax.
        We follow TabDDPM and anneal τ exponentially with global step.
        """
        import torch, torch.nn.functional as F
        try:
            gstep = torch.distributed.get_rank()  # non-DDP falls back except
        except Exception:
            gstep = 0
        tau_sched = max(0.5, 1.0 * (0.999 ** gstep))

        t = torch.tensor(logits)
        return F.gumbel_softmax(t, tau=tau_sched, hard=True).cpu().numpy()

CAT_MISSING = "__nan__"
EPS = 1e-5

def safe_log1p(x: np.ndarray) -> np.ndarray:        # clip negative drift
    return np.log1p(np.clip(x, 0., None))

def safe_logit(x: np.ndarray) -> np.ndarray:        # clip away from 0/1
    return logit(np.clip(x, EPS, 1.-EPS))

# ---------- helper that *is* picklable -------------------------------
def _identity(x):
    """Return input unchanged (picklable)."""
    return x


@dataclass
class NumSpec:
    """
    Meta information that allows inverse_transform & post-sampling repair.
    `scaler` is whatever the *last* step in the pipeline is, so we always
    have .inverse_transform available.
    """
    kind   : Literal["std", "log1p", "logit", "qt"]
    scaler : Any                    # StandardScaler | MinMaxScaler | QT
    min: Optional[float] = None
    max: Optional[float] = None
    is_int : bool = False           # round after inverse if True

def _build_num_pipeline(meta: dict) -> Tuple[Pipeline, NumSpec]:
    """
    Creates a scikit-learn Pipeline + NumSpec for *one* numeric column.
    Three cases (A,B,C) follow TabDDPM / TabDiff.
    """
    is_int = meta.get("sign") == "integer"
    lo, hi = meta.get("min"), meta.get("max")

    # (A) non-negative & unbounded  → clip-log1p-Standard
    if meta.get("sign") == "non-negative" and hi is None:
        pipe = Pipeline([
            ("clip0", FunctionTransformer(lambda z: np.clip(z, 0., None),
                                          validate=False)),
            ("log1p", FunctionTransformer(safe_log1p, np.expm1,
                                          validate=False)),
            ("std",   StandardScaler())
        ])
        return pipe, NumSpec("log1p", pipe["std"], is_int=is_int)

    # (B) bounded [lo,hi]  → MinMax→logit      (TabDiff style)
    _identity_tf = FunctionTransformer(lambda x: x, validate=False)
    if lo is not None and hi is not None:
        pipe = Pipeline([
            ("mm", MinMaxScaler(feature_range=(EPS, 1. - EPS), clip=True)),
            ("logit", FunctionTransformer(safe_logit, expit, validate=False)),
        ])
        # keep an identity "scaler" that *can* be pickled
        id_tf = FunctionTransformer(_identity, validate=False)
        return pipe, NumSpec("logit", id_tf, lo, hi, is_int)

    # (C) everything else  → Quantile→Normal
    qt = QuantileTransformer(output_distribution="normal",
                             n_quantiles=1024, subsample=int(1e6))
    pipe = Pipeline([("qt", qt)])
    # keep an identity "scaler" so .inverse_transform still works
    return pipe, NumSpec("qt", qt, is_int=is_int)

# -------------------------------------------------------------

@dataclass
class FittedTransforms:
    num_pipes   : Dict[str, Pipeline]
    num_specs   : Dict[str, NumSpec]
    cat_encoder : OneHotEncoder
    num_features: List[str]
    cat_features: List[str]
    cat_dims    : List[int]
    feature_list: List[str]
    cat_mode: Literal["onehot", "logits", "bits"] = "logits" #"onehot"
    # for "bits" we store bit-width per feature
    cat_bits: Optional[List[int]] = None

    # ------------- I/O convenience -----------------------------------
    def dump(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "FittedTransforms":
        with open(path, "rb") as f:
            return pickle.load(f)

# ----------------------------------------------------------------------
def fit_on_dataframe(df: pd.DataFrame, meta: List[dict]) -> FittedTransforms:
    df = (df.replace(9999.999, np.nan).replace(999.9999, np.nan))

    num_pipes, num_specs, nums, cats = {}, {}, [], []

    for m in meta:
        name = m["feature"]
        if m.get("dtype") == "categorical":
            cats.append(name)
        else:
            nums.append(name)
            pipe, spec = _build_num_pipeline(m)
            pipe = Pipeline([("imp", SimpleImputer(strategy="mean")),
                             *pipe.steps])
            pipe.fit(df[[name]].values)
            num_pipes[name], num_specs[name] = pipe, spec

    try:
        cat_encoder = OneHotEncoder(dtype=np.float32,
                                    handle_unknown='ignore',
                                    sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        cat_encoder = OneHotEncoder(dtype=np.float32,
                                    handle_unknown='ignore',
                                    sparse=False)
    if cats:
        cat_encoder.fit(df[cats].fillna(CAT_MISSING).values)

    return FittedTransforms(num_pipes, num_specs, cat_encoder,
                            nums, cats,
                            [len(c) for c in cat_encoder.categories_],
                            [m["feature"] for m in meta])

# ----------------------------------------------------------------------
def forward_transform(ft: FittedTransforms, row: np.ndarray) -> np.ndarray:
    assert row.ndim == 1
    out: list[np.ndarray] = []

    # --- categorical --------------------------------------------------
    if ft.cat_features:
        cat_vals = []
        for col in ft.cat_features:
            idx = ft.feature_list.index(col)
            val = row[idx]
            if np.isnan(val):
                val = CAT_MISSING
            cat_vals.append(val)
        onehot = ft.cat_encoder.transform([cat_vals]).ravel()
        if ft.cat_mode == "logits":
            encs, start = [], 0
            for k in ft.cat_dims:  # slice per feature
                oh_slice = onehot[start:start + k]
                encs.append(_CatLogits.fwd(oh_slice))
                start += k
            cat_enc = np.concatenate(encs, dtype=np.float32)
        elif ft.cat_mode == "bits":
            # pack each feature into binary code
            encs = []
            start = 0
            for k_i, k in enumerate(ft.cat_dims):
                code = onehot[start:start + k].argmax()
                w = ft.cat_bits[k_i]
                bits = np.unpackbits(np.array([code], np.uint8), bitorder="little")[: w]
                encs.append(bits.astype(np.float32))
                start += k
            cat_enc = np.concatenate(encs, dtype=np.float32)
        else:
            cat_enc = onehot

        out.append(cat_enc)

    # --- numeric ------------------------------------------------------
    for col in ft.num_features:
        v = row[ft.feature_list.index(col)]
        out.append(ft.num_pipes[col].transform([[v]]).ravel())

    return np.concatenate(out, dtype=np.float32)

# ----------------------------------------------------------------------
def inverse_transform(ft: FittedTransforms, x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x[None]
    B, n_cat = x.shape[0], sum(ft.cat_dims)
    cat_part, num_part = x[:, :n_cat], x[:, n_cat:]

    raw = np.empty((B, len(ft.feature_list)), np.float32)

    # numeric
    for i, col in enumerate(ft.num_features):
        spec, z = ft.num_specs[col], num_part[:, [i]]
        if spec.kind == "log1p":
            v = np.expm1(spec.scaler.inverse_transform(z))
        elif spec.kind == "logit":
            v01 = expit(spec.scaler.inverse_transform(z))
            v = spec.min + v01 * (spec.max - spec.min)
        elif spec.kind == "qt":
            v = spec.scaler.inverse_transform(z)
        else:  # "std"
            v = spec.scaler.inverse_transform(z)
        if spec.is_int:
            v = np.rint(v)
        raw[:, ft.feature_list.index(col)] = v.ravel()

    # categorical
    if ft.cat_features:
        if ft.cat_mode == "logits":
            one_hots, idx = [], 0
            for k in ft.cat_dims:
                log_slice = cat_part[:, idx:idx + k]
                one_hots.append(_CatLogits.inv(log_slice))
                idx += k
            oh = np.concatenate(one_hots, axis=1)
            cats = ft.cat_encoder.inverse_transform(oh)
        elif ft.cat_mode == "bits":
            codes, idx = [], 0
            for w in ft.cat_bits:
                bits = cat_part[:, idx:idx + w]
                code = np.packbits(bits.round().astype(np.uint8), bitorder="little")[:, 0]
                codes.append(code.reshape(-1, 1))
                idx += w
            cats = ft.cat_encoder.inverse_transform(np.concatenate(codes, 1))
        else:
            cats = ft.cat_encoder.inverse_transform(cat_part)
        for j, col in enumerate(ft.cat_features):
            raw[:, ft.feature_list.index(col)] = cats[:, j].astype(np.float32)
    return raw