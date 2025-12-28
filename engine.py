# ============================================
# engine.py - Motor Evaluación de Proyectos
# ============================================

# --------------------------------------------
# Imports
# --------------------------------------------
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np


# --------------------------------------------
# Modelo de datos
# --------------------------------------------
@dataclass
class ProjectInputs:
    name: str
    initial_investment: float      # CAPEX
    cash_flows: List[float]        # FCF año 1..N
    discount_rate: float           # WACC (decimal)
    terminal_value: float = 0.0
    currency: str = "CLP"


# --------------------------------------------
# Funciones financieras core
# --------------------------------------------
def npv(r: float, capex: float, cfs: List[float], tv: float = 0.0) -> float:
    pv = sum(float(cf) / (1.0 + float(r)) ** t for t, cf in enumerate(cfs, start=1))
    if tv != 0.0:
        pv += float(tv) / (1.0 + float(r)) ** len(cfs)
    return -float(capex) + pv


def irr(capex: float, cfs: List[float], tv: float = 0.0, guess: float = 0.15) -> Optional[float]:
    flows = [-float(capex)] + [float(x) for x in cfs]
    flows[-1] += float(tv)

    # chequeo mínimo de cambio de signo
    if not (min(flows) < 0 < max(flows)):
        return None

    r = float(guess)
    for _ in range(200):
        f, df = 0.0, 0.0
        for t, cf in enumerate(flows):
            f += cf / (1.0 + r) ** t
            if t > 0:
                df -= t * cf / (1.0 + r) ** (t + 1)

        if abs(df) < 1e-12:
            return None

        r_new = r - f / df
        if abs(r_new - r) < 1e-8:
            return r_new
        r = r_new

    return None


def payback(capex: float, cfs: List[float]) -> Optional[float]:
    remaining = float(capex)
    for year, cf in enumerate(cfs, start=1):
        cf = float(cf)
        if cf >= remaining:
            return (year - 1) + (remaining / cf)
        remaining -= cf
    return None


# --------------------------------------------
# Evaluación ejecutiva
# --------------------------------------------
def evaluate_project(p: ProjectInputs) -> Dict:
    van = npv(p.discount_rate, p.initial_investment, p.cash_flows, p.terminal_value)
    tir = irr(p.initial_investment, p.cash_flows, p.terminal_value)
    pb = payback(p.initial_investment, p.cash_flows)

    # Gate conservador: VAN>0 y si existe TIR, TIR>WACC
    decision = "APROBAR" if (van > 0 and (tir is None or tir > p.discount_rate)) else "RECHAZAR"

    return {
        "Proyecto": p.name,
        "Moneda": p.currency,
        "CAPEX": float(p.initial_investment),
        "WACC": float(p.discount_rate),
        "VAN": float(van),
        "TIR": None if tir is None else float(tir),
        "Payback": None if pb is None else float(pb),
        "Decisión": decision,
    }


# --------------------------------------------
# Escenarios estándar
# --------------------------------------------
def scenarios(p: ProjectInputs) -> Dict[str, Dict]:
    base = evaluate_project(p)

    optimistic = ProjectInputs(
        name=p.name + " (Optimista)",
        initial_investment=p.initial_investment,
        cash_flows=[float(cf) * 1.10 for cf in p.cash_flows],
        discount_rate=float(p.discount_rate) - 0.005,
        terminal_value=p.terminal_value,
        currency=p.currency,
    )

    stress = ProjectInputs(
        name=p.name + " (Estrés)",
        initial_investment=p.initial_investment,
        cash_flows=[float(cf) * 0.85 for cf in p.cash_flows],
        discount_rate=float(p.discount_rate) + 0.02,
        terminal_value=p.terminal_value,
        currency=p.currency,
    )

    return {
        "Base": base,
        "Optimista": evaluate_project(optimistic),
        "Estrés": evaluate_project(stress),
    }


# --------------------------------------------
# Monte Carlo (riesgo)
# --------------------------------------------
def monte_carlo(p: ProjectInputs, n: int = 3000, sigma_cf: float = 0.15, sigma_r: float = 0.01, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    npvs = []

    for _ in range(int(n)):
        cfs_sim = [float(cf) * rng.lognormal(mean=0.0, sigma=float(sigma_cf)) for cf in p.cash_flows]
        r_sim = max(-0.99, float(rng.normal(loc=float(p.discount_rate), scale=float(sigma_r))))
        npvs.append(npv(r_sim, p.initial_investment, cfs_sim, p.terminal_value))

    arr = np.array(npvs, dtype=float)
    return {
        "VAN Medio": float(arr.mean()),
        "P05": float(np.percentile(arr, 5)),
        "P50": float(np.percentile(arr, 50)),
        "P95": float(np.percentile(arr, 95)),
        "Prob VAN > 0": float((arr > 0).mean()),
    }


# --------------------------------------------
# Sensibilidad
# --------------------------------------------
def sensitivity_npv_vs_discount_rate(p: ProjectInputs, rate_grid: List[float]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for r in rate_grid:
        out.append((float(r), npv(float(r), p.initial_investment, p.cash_flows, p.terminal_value)))
    return out


def sensitivity_npv_vs_cf_multiplier(p: ProjectInputs, multipliers: List[float]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for m in multipliers:
        cfs_scaled = [float(cf) * float(m) for cf in p.cash_flows]
        van = npv(float(p.discount_rate), p.initial_investment, cfs_scaled, float(p.terminal_value) * float(m))
        out.append((float(m), float(van)))
    return out
