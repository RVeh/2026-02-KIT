from __future__ import annotations

from typing import List, Dict, Any

from core.ci_methods_core import all_ci_methods, CIResult


def show_ci_methods(n: int, k: int, gamma: float) -> List[Dict[str, Any]]:
    """
    SuS/LuL-freundlicher Einstieg:
    Gibt eine kleine Tabelle (Liste von Dicts) zurück:
      Methode | L | U | Breite
    """
    results = all_ci_methods(k=k, n=n, gamma=gamma)

    rows: List[Dict[str, Any]] = []
    for name in ["Wald", "Wilson", "Clopper–Pearson"]:
        r: CIResult = results[name]
        rows.append(
            {
                "Methode": r.method,
                "L": r.L,
                "U": r.U,
                "Breite": r.width,
            }
        )
    return rows
