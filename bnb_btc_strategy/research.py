"""Optional, *offline* statistical diagnostics; never auto-trade from tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def johansen_diagnostic(btc_close: pd.Series, bnb_close: pd.Series,
                        lookback: int = 256) -> dict[str, float | bool]:
    """Johansen trace diagnostic on PAST matched closes; not a trade forecast.

    Raises rather than reporting a fabricated result for insufficient samples,
    missing observations or a missing optional statsmodels installation.
    """
    if lookback < 64:
        raise ValueError("lookback must be >= 64")
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    samples = pd.concat([btc_close.rename("btc"), bnb_close.rename("bnb")],
                        axis=1).dropna().tail(lookback)
    if len(samples) < lookback or (samples <= 0).any().any():
        raise ValueError("need positive, matched past closes for entire lookback")
    logged = np.log(samples)
    if (logged.std() <= 1e-10).any():
        raise ValueError("constant series cannot be tested")
    result = coint_johansen(logged, det_order=0, k_ar_diff=1)
    trace = float(result.lr1[0])
    critical_95 = float(result.cvt[0, 1])
    return {"trace_statistic": trace, "critical_95": critical_95,
            "reject_rank_zero_95": bool(trace > critical_95)}
