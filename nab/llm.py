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
from properscoring import crps_ensemble        #for LL‑score ⇒ anomaly

from lib.python3_8.site_packages.rcf.trcf_model import TRandomCutForestModel as TRCF


# ----------------------------------------------
# 1.  NAB helpers  (edit ONLY this first block)
# ----------------------------------------------
from pathlib import Path
import pandas as pd
from typing import Iterable, Tuple, List, Dict

CSV_DIR = Path("./")

# ✦  STEP A – choose the five streams ✦
SELECTED_STREAMS: set[str] = {
    "ec2_cpu_utilization_24ae8d.csv",
    "ec2_network_in_257a54.csv",
    "ec2_disk_write_bytes_1ef3de.csv",
    "rds_cpu_utilization_e47b3b.csv",
    "rds_cpu_utilization_cc0c53.csv",
}

# ✦  STEP B – place the anomaly‑window *starts* here ✦
MANUAL_LABELS: Dict[str, List[str]] = {
    "ec2_cpu_utilization_24ae8d.csv": [
        "2014-02-26 22:05:00",
        "2014-02-27 17:15:00",
    ],
    "ec2_network_in_257a54.csv": [
        "2014-04-15 16:44:00",
    ],
    "ec2_disk_write_bytes_1ef3de.csv": [
        "2014-03-10 21:09:00",
    ],
    "rds_cpu_utilization_e47b3b.csv": [
        "2014-04-13 06:52:00",
        "2014-04-18 23:27:00",
    ],
    "rds_cpu_utilization_cc0c53.csv": [
        "2014-02-25 07:15:00",
        "2014-02-27 00:50:00",
    ],
}

def load_nab_streams(
    limit: set[str] = SELECTED_STREAMS,
    labels: Dict[str, List[str]] = MANUAL_LABELS,
) -> Iterable[Tuple[str, pd.DataFrame, List[pd.Timestamp]]]:
    """
    Yield (stream_name, df, label_ts) **only** for the CSV files in *limit*.
    `label_ts` is a list of pd.Timestamp objects built from MANUAL_LABELS.
    """
    for csv_path in CSV_DIR.glob("**/*.csv"):
        name = csv_path.name
        if name not in limit:
            continue

        df = pd.read_csv(csv_path, names=["timestamp", "value"], header=0, skiprows=1)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        label_ts = [pd.to_datetime(s) for s in labels.get(name, [])]
        yield name, df, label_ts


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
        return desc.getAnomalyGrade()


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
from toto.data.util.dataset import MaskedTimeseries           # NEW

class TotoDetector(BaseDetector):
    def __init__(self, device=None, **kw):
        super().__init__(**kw)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        core = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0")\
                   .to(self.device).eval()
        self.model = core                       # so .parameters() works
        self.fcst  = TotoForecaster(core.model)

        self.ctx: List[float] = []

    def update(self, ts, value):
        """
        Build an ad‑hoc 1‑channel `MaskedTimeseries`, ask Toto for a
        one‑step‑ahead *median* forecast, and use |actual‑median| as the
        anomaly score (larger = more anomalous).
        """
        self.ctx.append(value)
        if len(self.ctx) < 64:          # warm‑up window
            return 0.0

        # keep the last 512 points, shape = (channels, time_steps)
        series = torch.tensor(self.ctx[-512:], dtype=torch.float32,
                              device=self.device).unsqueeze(0)

        dummy_mask = torch.ones_like(series, dtype=torch.bool)
        ts_inp = MaskedTimeseries(
            series          = series,
            padding_mask    = dummy_mask,
            id_mask         = torch.zeros_like(series),
            timestamp_seconds = torch.zeros_like(series),
            time_interval_seconds = torch.full((1,), 60, device=self.device),
        )

        fc = self.fcst.forecast(
            ts_inp, prediction_length=1,
            num_samples=128, samples_per_batch=128,
        )
        μ = fc.median.item()
        # 84.1% of the mass lies below μ+1σ.
        σ = fc.quantile(0.84).item() - μ
        return abs(value - μ) / (σ + 1e-9)


# ---------------- 2‑D  Chronos‑T5  (z‑score) --------------------------
from collections import deque

class ChronosDetector(BaseDetector):
    """
    Uses Chronos‑T5‑small for one‑step prediction and converts the absolute
    error to *z = |err| / σ̂* where σ̂ is the running std‑dev of recent errors.
    """
    _CTX      = 256      # last points fed to the model
    _WARM_UP  = 64       # points before we start emitting scores
    _ERR_WIN  = 512      # window for std‑dev estimate

    def __init__(self, **kw):
        super().__init__(**kw)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        from chronos import BaseChronosPipeline

        # This loads *both* the HF model *and* the numeric tokenizer.
        pipe = BaseChronosPipeline.from_pretrained(
                    "amazon/chronos-t5-small",
                    device_map = None,  # we move it manually next line
                    torch_dtype = torch.float32,
        )

        self.tok = pipe.tokenizer  # MeanScaleUniformBins
        self.model = pipe.model.to(self.device).eval()  # T5‑style LM

        self.ctx : List[float]    = []
        self.err_buf = deque(maxlen=self._ERR_WIN)   # store recent |errors|

    def update(self, ts, value: float) -> float:
        self.ctx.append(value)
        if len(self.ctx) < self._WARM_UP:
            return 0.0

        # Build a numeric context tensor
        ctx = torch.tensor(self.ctx[-self._CTX:], dtype=torch.float32,
                           device=self.device).unsqueeze(0)

        # Convert context to token IDs, attention mask and scale
        token_ids, attention_mask, scale = self.tok.context_input_transform(ctx)
        token_ids = token_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        scale = scale.to(self.device)

        # Generate the next token
        with torch.no_grad():
            out = self.model.generate(input_ids=token_ids,
                                      attention_mask=attention_mask,
                                      max_new_tokens=1)

        # Take the new token (last position) and add a fake num_samples dimension
        new_token = out[:, -1:].unsqueeze(1)  # shape (1, 1, 1)

        # Decode token back to a real value using output_transform
        pred = self.tok.output_transform(new_token, scale)[0, 0, 0].item()

        # Use the absolute error to compute an anomaly score with running std‑dev
        err = abs(value - pred)
        self.err_buf.append(err)
        sigma = np.std(self.err_buf) + 1e-9
        return err / sigma


# ---------------- 2‑E  TimesFM‑2.0‑500M ------------------------------
# import timesfm
# class TimesFMDetector(BaseDetector):
#     def __init__(self, **kw):
#         super().__init__(**kw)
#         self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         self.model = timesfm.load_from_hf_hub("google/timesfm-2.0-500m-pytorch")\
#                         .to(self.device).eval()
#         self.ctx: List[float] = []
#     def update(self, ts, value):
#         self.ctx.append(value)
#         if len(self.ctx) < 64:
#             return 0
#         mean, std = self.model.forecast_mean_std(self.ctx[-512:], horizon=1)
#         return zscore(mean.item(), std.item(), value)s


# ---------------- 2‑F  Moirai‑Large  (z‑score) -----------------------
from uni2ts.model.moirai import MoiraiModule

class MoiraiDetector(BaseDetector):
    """
    Moirai returns a Distribution object ⇒ we can take its mean & stddev
    directly and compute a true z‑score: z = |y − μ| / σ.
    """
    _CTX      = 400
    _WARM_UP  = 128

    def __init__(self, **kw):
        super().__init__(**kw)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model  = MoiraiModule.from_pretrained(
                          "Salesforce/moirai-1.0-R-large").to(self.device).eval()
        self.ctx: List[float] = []

    def update(self, ts, value: float) -> float:
        self.ctx.append(value)
        if len(self.ctx) < self._WARM_UP:
            return 0.0

        ctx_t = torch.tensor(self.ctx[-self._CTX:], dtype=torch.float32,
                             device=self.device)[None, :, None]

        with torch.no_grad():
            dist = self.model(ctx_t, horizon=1)["dist"]   # torch.distributions

        mu   = dist.mean.squeeze().item()
        sigma= dist.stddev.squeeze().item() + 1e-9        # avoid div‑by‑0
        return abs(value - mu) / sigma


# --------------- 2‑G  PatchTST, TTM‑R2, TEMPO, TabPFN‑TS -------------
# (identical pattern – omitted for brevity, add if you need all models)
# ---------------------------------------------------------------------


DETECTORS = {
    "RCF"     : lambda: RCFDetector(),            # grade → use 0
    # "Toto"    : lambda: TotoDetector(),
    "Chronos" : lambda: ChronosDetector(),
    "Moirai"  : lambda: MoiraiDetector(),
}

# The evaluate() driver originally expected every detector to emit a “large‑means‑bad” score (e.g., a z‑score or –logp).
# A convenient, if blunt, rule of thumb is “flag anything >≈3σ”, hence the hard‑coded default threshold=3.0.
THRESHOLDS = {
    "RCF"    : 0.0,       # any positive grade
    # "Toto"   : 3.0,       # –logp or z-score style
    "Chronos": 3.0,
    "Moirai" : 3.0,
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
            p,r = stream_metrics(scores, label_ts, ts_list, THRESHOLDS[name])
            results[name]["tp"] += r * len(label_ts)      # coarse but fine for macro avg
            results[name]["fn"] += (1-r) * len(label_ts)
            if p:  # avoid div‑by‑zero
                fp = r * len(label_ts) * (1 / p - 1)
                results[name]["fp"] += fp
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
