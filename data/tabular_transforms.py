import json, pickle, math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Literal, Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer, QuantileTransformer
)
from sklearn.impute import SimpleImputer


class _CatLogits:
    """
    Utility to convert one-hot ⇄ logits per *single* feature.
    EPS = 1e-3  # gentler tails to avoid ±9.2 scales
    """
    EPS = 1e-4
    @staticmethod
    def fwd(oh: np.ndarray) -> np.ndarray:
        """
        one-hot → tempered logits (label-smoothed):
        p = (1-τ)*onehot + τ/k,  τ≈0.05..0.1
        Then map to per-dim logit(p) to keep magnitudes moderate.
        """

        k = oh.shape[-1]
        # gentler smoothing to avoid over‑compression of logits when k is small
        tau = min(0.05, 1.0 / (k + 10))
        p = (1.0 - tau) * oh + (tau / k)

        return logit(np.clip(p, _CatLogits.EPS, 1 - _CatLogits.EPS))

    @staticmethod
    def inv(logits: np.ndarray) -> np.ndarray:
        """
        logits → one‑hot (deterministic argmax).
        Decoding must be noise‑free and strictly invertible at sample time.
        """

        l = logits

        if l.ndim == 1:
            l = l[None]
        idx = np.argmax(l, axis=-1)
        oh = np.zeros_like(l, dtype=np.float32)
        oh[np.arange(l.shape[0]), idx] = 1.0

        return oh

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
    min    : float | None = None
    max    : float | None = None
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
            ("clip0", FunctionTransformer(lambda z: np.clip(z, 0.0, None),
            feature_names_out="one-to-one")),
            ("log1p", FunctionTransformer(safe_log1p, np.expm1,
                                          validate=False)),
            ("std",   StandardScaler())
        ])
        return pipe, NumSpec("log1p", pipe["std"], is_int=is_int)

    # (B) bounded [lo,hi] → MinMax([0,1]) → safe_logit → StandardScaler
    if lo is not None and hi is not None:
        mm = MinMaxScaler(feature_range=(EPS, 1.0 - EPS))
        pipe = Pipeline([
            ("minmax", mm),
            ("logit", FunctionTransformer(safe_logit, expit, validate=False)),
            ("std", StandardScaler())
        ])
        return pipe, NumSpec("logit", pipe["std"], lo, hi, is_int)

    # (C) everything else  → Quantile→Normal
    qt = QuantileTransformer(output_distribution="normal",
                             n_quantiles=2048, subsample=int(1e6))
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
    cat_bits: List[int] | None = None

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

    cat_encoder = OneHotEncoder(dtype=np.float32,
                                handle_unknown='ignore',
                                sparse_output=False)
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
            cat_enc = np.concatenate(encs).astype(np.float32)
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
            cat_enc = np.concatenate(encs).astype(np.float32)
        else:
            cat_enc = onehot

        out.append(cat_enc)

    # --- numeric ------------------------------------------------------
    for col in ft.num_features:
        v = row[ft.feature_list.index(col)]
        out.append(ft.num_pipes[col].transform([[v]]).ravel())

    return np.concatenate(out).astype(np.float32)

# ----------------------------------------------------------------------
def inverse_transform(ft: FittedTransforms, x: np.ndarray, *, return_object: bool = False) -> np.ndarray:
    if x.ndim == 1:
        x = x[None]
    B, n_cat = x.shape[0], sum(ft.cat_dims)
    cat_part, num_part = x[:, :n_cat], x[:, n_cat:]

    # (4.c) If return_object=True, keep categorical strings (object dtype);
    # fallback to previous float32 behavior otherwise (backward compatible).
    if return_object:
        raw = np.empty((B, len(ft.feature_list)), dtype=object)
    else:
        raw = np.empty((B, len(ft.feature_list)), np.float32)

    # numeric
    for i, col in enumerate(ft.num_features):
        spec, z = ft.num_specs[col], num_part[:, [i]]
        if spec.kind == "log1p":
            v = np.expm1(spec.scaler.inverse_transform(z))
        elif spec.kind == "logit":
            # Undo: StandardScaler ⟶ logit ⟶ MinMax([EPS, 1−EPS])
            p = expit(spec.scaler.inverse_transform(z))
            p = np.clip(p, EPS, 1.0 - EPS)  # enforce valid range
            v = spec.min + (p - EPS) * (spec.max - spec.min) / (1.0 - 2.0 * EPS)
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
            if return_object:
                # preserve original (possibly string) categories
                raw[:, ft.feature_list.index(col)] = cats[:, j]
            else:
                # legacy behavior (numeric casting)
                raw[:, ft.feature_list.index(col)] = cats[:, j].astype(np.float32)
    return raw