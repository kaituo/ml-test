"""
run_nab_benchmark.py
Compares RCF with open‑weights TSFMs on NAB (precision/recall + cost/latency)

> pip install -U pandas numpy torch tqdm properscoring \
              transformers==4.* gluonts==0.14.3 \
              timesfm uni2ts tabpfn tempo patchtst ttm-r2

Clone NAB once:
    git clone https://github.com/numenta/NAB.git
"""

from __future__ import annotations
import json, os, time, math, argparse, gc, psutil
from collections import deque, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Iterable

import numpy as np, pandas as pd, torch
from sklearn.metrics import precision_score, recall_score
from properscoring import crps_ensemble        # for LL‑score ⇒ anomaly

from lib.python3_8.site_packages.rcf.trcf_model import TRandomCutForestModel as TRCF


# ---------------------------------------------------------------------
# 1.  NAB helpers
# ---------------------------------------------------------------------
NAB_ROOT   = Path(os.getenv("NAB_ROOT", "./NAB"))
CSV_DIR    = NAB_ROOT / "data"
LABEL_FILE = NAB_ROOT / "labels" / "combined_labels.json"
with open(LABEL_FILE) as fh:
    RAW_LABELS: Dict[str, List[List[str]]] = json.load(fh)          # {file: [[start,end],…]}

def load_nab_streams() -> Iterable[Tuple[str, pd.DataFrame, List[pd.Timestamp]]]:
    """
    Yields (stream_name, df, label_timestamps) where df has timestamp,value columns.
    Labels are converted to the *start* of each anomaly window (acceptable for
    5‑minute matching window).
    """
    for csv_path in CSV_DIR.glob("**/*.csv"):
        name = csv_path.name
        df   = pd.read_csv(csv_path)
        df.columns = ["timestamp", "value"]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        labels = [pd.to_datetime(w[0]) for w in RAW_LABELS.get(name, [])]
        yield name, df, labels


# ---------------------------------------------------------------------
# 2.  Base detector interface  ----------------------------------------
# ---------------------------------------------------------------------
class BaseDetector:
    """Streaming interface every model implements."""
    def __init__(self, threshold: float = 3.0):
        self.th = threshold
    def update(self, ts: pd.Timestamp, value: float) -> float:
        """Return anomaly *score* (higher == more anomalous)."""
        raise NotImplementedError


# ---------------- 2‑A  RCF wrapper -----------------------------------
class RCFDetector(BaseDetector):
    def __init__(self, shingle=8, num_trees=50, **kw):
        super().__init__(**kw)
        self.shingle = shingle
        self.rcf = TRCF(rcf_dimensions=shingle,
                        shingle_size=shingle,
                        num_trees=num_trees,
                        output_after=32,
                        anomaly_rate=0.005,
                        z_factor=3)
    def update(self, ts, value):
        desc = self.rcf.process(np.array([value], dtype="float64"), 0)
        return desc.getAnomalyScore()


# ---------------- 2‑B  Helper to turn *forecast* into anomaly score ---
def nll_score(dist, y: torch.Tensor):
    """Negative log‑likelihood for one value; dist is a torch.distribution."""
    return -dist.log_prob(y).item()

def zscore(mean, std, y):
    return abs(y - mean) / (std + 1e-9)

NUM_SAMPLES = 128        # used for CRPS fallback


# ---------------- 2‑C  Toto ------------------------------------------
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto
class TotoDetector(BaseDetector):
    def __init__(self, device=None, **kw):
        super().__init__(**kw)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        m = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0").to(self.device).eval()
        self.fcst = TotoForecaster(m.model)
        self.ctx  : List[float] = []
    def update(self, ts, value):
        self.ctx.append(value)
        if len(self.ctx) < 64:           # warm‑up
            return 0.0
        ctx = torch.tensor(self.ctx[-512:], dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.fcst.predictive_distribution(ctx)          # Student‑T mixture
        return nll_score(dist, torch.tensor(value, device=self.device))


# ---------------- 2‑D  Chronos‑T5 ------------------------------------
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
class ChronosDetector(BaseDetector):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model  = AutoModelForSeq2SeqLM.from_pretrained("amazon/chronos-t5-small")\
                        .to(self.device).eval()
        self.tok    = AutoTokenizer.from_pretrained("amazon/chronos-t5-small")
        self.ctx: List[float] = []
    def update(self, ts, value):
        self.ctx.append(value)
        if len(self.ctx) < 32:
            return 0
        #  Chronos tokenises the series as space‑separated ints; keep last 256 pts
        prompt = " ".join(f"{v:.4f}" for v in self.ctx[-256:])
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=1)
        pred = float(self.tok.decode(out[0], skip_special_tokens=True).split()[-1])
        return abs(value - pred)


# ---------------- 2‑E  TimesFM‑2.0‑500M ------------------------------
import timesfm
class TimesFMDetector(BaseDetector):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = timesfm.load_from_hf_hub("google/timesfm-2.0-500m-pytorch")\
                        .to(self.device).eval()
        self.ctx: List[float] = []
    def update(self, ts, value):
        self.ctx.append(value)
        if len(self.ctx) < 64:
            return 0
        mean, std = self.model.forecast_mean_std(self.ctx[-512:], horizon=1)
        return zscore(mean.item(), std.item(), value)


# ---------------- 2‑F  Moirai‑Large ----------------------------------
from uni2ts.model.moirai import MoiraiModule
class MoiraiDetector(BaseDetector):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.mod = MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-large")\
                     .to(self.device).eval()
        self.ctx = []
    def update(self, ts, value):
        self.ctx.append(value)
        if len(self.ctx) < 128:
            return 0
        ctx_t = torch.tensor(self.ctx[-400:], dtype=torch.float32,
                             device=self.device)[None, :, None]
        dist  = self.mod(ctx_t, horizon=1)["dist"]
        return nll_score(dist, torch.tensor(value, device=self.device))


# --------------- 2‑G  PatchTST, TTM‑R2, TEMPO, TabPFN‑TS -------------
# (identical pattern – omitted for brevity, add if you need all models)
# ---------------------------------------------------------------------


DETECTORS = {
    "RCF"      : lambda: RCFDetector(),
    "Toto"     : lambda: TotoDetector(),
    "Chronos"  : lambda: ChronosDetector(),
    "TimesFM"  : lambda: TimesFMDetector(),
    "Moirai"   : lambda: MoiraiDetector(),
    # "PatchTST" : lambda: PatchTSTDetector(),
    # "TTM-R2"   : lambda: TTMDetector(),
    # "TEMPO"    : lambda: TempoDetector(),
    # "TabPFN"   : lambda: TabPFNDetector(),
}


# ---------------------------------------------------------------------
# 3.  Evaluation helpers (precision/recall & latency)
# ---------------------------------------------------------------------
WINDOW_MINUTES = 5
def match_window(ts1, ts2, win=pd.Timedelta(minutes=WINDOW_MINUTES)) -> bool:
    return abs(ts1 - ts2) <= win

def stream_metrics(scores: List[float], labels: List[pd.Timestamp],
                   ts_list: List[pd.Timestamp], thresh: float):
    """Return precision, recall for one stream given *scalar* scores."""
    preds = [s > thresh for s in scores]
    # create label bools by marking any ts within ±win of an anomaly start
    y_true = [any(match_window(t,l) for l in labels) for t in ts_list]
    if not any(y_true):
        return 0,0
    precision = precision_score(y_true, preds)
    recall    = recall_score(y_true, preds)
    return precision, recall


# ---------------------------------------------------------------------
# 4.  Main benchmarking loop
# ---------------------------------------------------------------------
def evaluate(threshold=3.0):
    results = defaultdict(lambda: {"tp":0,"fp":0,"fn":0,
                                   "lat_sum":0.0,"cnt":0,
                                   "params":0,"mem":0})
    streams = list(load_nab_streams())
    for name, init_fn in DETECTORS.items():
        det: BaseDetector = init_fn()
        # capture cost proxy
        if hasattr(det, "model"):
            m = det.model
        elif hasattr(det, "mod"):
            m = det.mod
        elif hasattr(det, "fcst"):
            m = det.fcst
        else:
            m = None
        params = sum(p.numel() for p in m.parameters())/1e6 if m else 0
        mem    = torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else \
                 psutil.Process(os.getpid()).memory_info().rss/1e9
        results[name]["params"] = params
        results[name]["mem"]    = max(results[name]["mem"], mem)

        for stream_name, df, label_ts in streams:
            scores, ts_list = [], []
            for ts, val in zip(df["timestamp"], df["value"]):
                t0 = time.perf_counter()
                s  = det.update(ts, float(val))
                results[name]["lat_sum"] += (time.perf_counter() - t0)
                results[name]["cnt"]     += 1
                scores.append(s)
                ts_list.append(ts)
            p,r = stream_metrics(scores, label_ts, ts_list, threshold)
            results[name]["tp"] += p * len(label_ts)      # coarse but fine for macro avg
            results[name]["fn"] += (1-r) * len(label_ts)
            results[name]["fp"] += max(0, round(p*len(label_ts)/r) - p*len(label_ts)) if r else 0
            gc.collect(); torch.cuda.empty_cache()

    # Summarise
    rows = []
    for n,d in results.items():
        prec = d["tp"]/(d["tp"]+d["fp"]+1e-9)
        rec  = d["tp"]/(d["tp"]+d["fn"]+1e-9)
        lat  = 1e3 * d["lat_sum"]/d["cnt"]      # ms
        rows.append((n, prec, rec, lat, d["params"], d["mem"]))
    return pd.DataFrame(rows, columns=["Model","Precision","Recall",
                                       "Avg ms/pt","Params (M)","Peak GB"])

if __name__ == "__main__":
    df = evaluate(threshold=3.0)
    print(df.sort_values("Precision", ascending=False).to_string(index=False))
