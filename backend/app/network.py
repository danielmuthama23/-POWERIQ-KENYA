"""Explicit transmission network model with DC power-flow (linearised) approximation.

The network is an ILLUSTRATIVE ~18-bus model of Kenya's backbone. Line reactances are proportional to
route length and capacities are assumed. This is a DC approximation for bottleneck screening only; it is
NOT a validated AC power-flow study and must not be used for operational decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SLACK = "EMBAKASI"


class DCNetwork:
    def __init__(self, buses: pd.DataFrame, lines: pd.DataFrame):
        self.buses = list(buses.bus_id)
        self.idx = {b: i for i, b in enumerate(self.buses)}
        self.lines = lines.reset_index(drop=True)
        n, m = len(self.buses), len(self.lines)
        A = np.zeros((m, n))
        for k, r in self.lines.iterrows():
            A[k, self.idx[r.from_bus]] = 1
            A[k, self.idx[r.to_bus]] = -1
        b = 1.0 / self.lines.x_pu.values
        self.A, self.b = A, b
        Bbus = A.T @ np.diag(b) @ A
        keep = [i for i, bus in enumerate(self.buses) if bus != SLACK]
        self.keep = keep
        Bred = Bbus[np.ix_(keep, keep)]
        self.Xred = np.linalg.inv(Bred)  # raises if the network is disconnected
        # PTDF: flow_k = sum_i PTDF[k,i] * injection_i (slack absorbs imbalance)
        self.ptdf = np.zeros((m, n))
        self.ptdf[:, keep] = (np.diag(b) @ A[:, keep]) @ self.Xred
        self.cap = self.lines.capacity_mw.values.astype(float)

    def flows(self, injections: np.ndarray, derate: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """injections: [T, n_buses] MW (generation - load). Returns flows [T, m] and loading ratio."""
        inj = np.asarray(injections, float)
        inj = inj - np.where(np.arange(len(self.buses)) == self.idx[SLACK], inj.sum(axis=1, keepdims=True), 0.0)
        f = inj @ self.ptdf.T
        cap = self.cap * (1.0 if derate is None else derate)
        return f, np.abs(f) / np.maximum(cap, 1e-9)

    def bus_vector(self, mapping: dict[str, float]) -> np.ndarray:
        v = np.zeros(len(self.buses))
        for k, val in mapping.items():
            v[self.idx[k]] += val
        return v
