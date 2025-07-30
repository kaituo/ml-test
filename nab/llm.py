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
from typing import List, Dict, Tuple, Iterable, Optional

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

# ✦  STEP B – place the anomaly‑window *starts* here ✦
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

    def update(self, ts: pd.Timestamp, value: float) -> Tuple[float, Optional[pd.Timestamp]]:
        """Return (score, adjusted_timestamp or None)."""


# ---------------- 2‑A  RCF wrapper -----------------------------------
class RCFDetector(BaseDetector):
    def __init__(self, shingle=8, num_trees=50, **kw):
        super().__init__(**kw)
        self.rcf = TRCF(rcf_dimensions=shingle,
                        shingle_size=shingle,
                        num_trees=num_trees,
                        output_after=32,
                        anomaly_rate=0.005,
                        z_factor=3,
                        score_differencing=0.5,
                        ignore_delta_threshold_ratio=0.2
                        )
        self._ts_buf: deque[pd.Timestamp] = deque(maxlen=shingle)

    def update(self, ts, value):
        desc = self.rcf.process(np.array([value], dtype="float64"), 0)
        grade = desc.getAnomalyGrade()

        # push *after* calling process(), to mirror your perMinute logic
        self._ts_buf.append(ts)

        if grade <= self.th:
            return grade, None  # not an alert

        rel = desc.getRelativeIndex()  # 0 = "now", -1 = "prev", …
        if rel == 0:
            adj_ts = ts
        else:
            # deque[-1] is *current* ts, so shift by -1
            idx = rel - 1
            try:
                adj_ts = self._ts_buf[idx]
            except IndexError:
                adj_ts = ts  # fallback: use current ts
        return grade, adj_ts


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
    # ─── class‑level cache ──────────────────────────────────────
    _core = None  # Toto base model (with .model attribute)
    _fcst = None  # TotoForecaster wrapping _core.model

    def __init__(self, device=None, **kw):
        super().__init__(**kw)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        # load once, share for all instances
        if TotoDetector._core is None:
            core = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0").eval()
            TotoDetector._core = core
            TotoDetector._fcst = TotoForecaster(core.model)

        # bind the shared objects to *this* instance
        self.model = TotoDetector._core.to(self.device)  # for param counting
        self.fcst = TotoDetector._fcst

        self.ctx: List[float] = []

    def update(self, ts, value):
        """
        Build an ad‑hoc 1‑channel `MaskedTimeseries`, ask Toto for a
        one‑step‑ahead *median* forecast, and use |actual‑median| as the
        anomaly score (larger=more anomalous).
        """
        self.ctx.append(value)
        if len(self.ctx) < 64:          # warm‑up window
            return 0.0, None

        # keep the last 512 points, shape=(channels, time_steps)
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
        grade = abs(value - μ) / (σ + 1e-9)
        if grade <= self.th:
            return grade, None
        return grade, ts
#
#
# # ---------------- 2‑D  Chronos‑T5  (z‑score) --------------------------
from collections import deque

class ChronosDetector(BaseDetector):
    """
    Uses Chronos‑T5‑small for one‑step prediction and converts the absolute
    error to *z = |err| / σ̂* where σ̂ is the running std‑dev of recent errors.
    """
    # ─── class‑level singletons ───────────────────────────────
    _tok = None  # tokenizer
    _model = None  # HF T5 model (on the right device)

    _CTX      = 256      # last points fed to the model
    _WARM_UP  = 64       # points before we start emitting scores
    _ERR_WIN  = 512      # window for std‑dev estimate

    def __init__(self, **kw):
        super().__init__(**kw)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if ChronosDetector._model is None:
            from chronos import BaseChronosPipeline
            pipe = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-t5-small",
                device_map=None, # we move it manually
                torch_dtype=torch.float32)
            ChronosDetector._tok = pipe.tokenizer
            ChronosDetector._model = pipe.model.to(self.device).eval()

        # all instances (probe + per‑CSV) share the same objects
        self.tok = ChronosDetector._tok
        self.model = ChronosDetector._model

        # per‑instance rolling buffers
        self.ctx : List[float]    = []
        self.err_buf = deque(maxlen=self._ERR_WIN)   # store recent |errors|

    def update(self, ts, value: float) -> float:
        self.ctx.append(value)
        if len(self.ctx) < self._WARM_UP:
            return 0.0, None

        # Build context on CPU
        ctx = torch.tensor(self.ctx[-self._CTX:], dtype=torch.float32).unsqueeze(0)
        token_ids, attention_mask, scale = self.tok.context_input_transform(ctx)

        # Move IDs and masks to the device
        token_ids = token_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Generate the next token using the *underlying* HF model
        with torch.no_grad():
            out = self.model.model.generate(  # note: .model.generate(...)
                input_ids=token_ids,
                attention_mask=attention_mask,
                max_new_tokens=1
            )

        new_token = out[:, -1:].unsqueeze(1).to('cpu')
        pred = self.tok.output_transform(new_token, scale)[0, 0, 0].item()

        err = abs(value - pred)
        self.err_buf.append(err)
        sigma = np.std(self.err_buf) + 1e-9
        grade = err / sigma
        if grade <= self.th:
            return grade, None
        return grade, ts


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
# from uni2ts.model.moirai import MoiraiModule
#
# class MoiraiDetector(BaseDetector):
#     """
#     Moirai returns a Distribution object ⇒ we can take its mean & stddev
#     directly and compute a true z‑score: z = |y − μ| / σ.
#     """
#     _CTX      = 400
#     _WARM_UP  = 128
#
#     def __init__(self, **kw):
#         super().__init__(**kw)
#         self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         self.model  = MoiraiModule.from_pretrained(
#                           "Salesforce/moirai-1.0-R-large").to(self.device).eval()
#         self.ctx: List[float] = []
#
#     def update(self, ts, value: float) -> float:
#         self.ctx.append(value)
#         if len(self.ctx) < self._WARM_UP:
#             return 0.0
#
#         ctx_t = torch.tensor(self.ctx[-self._CTX:], dtype=torch.float32,
#                              device=self.device)[None, :, None]
#
#         with torch.no_grad():
#             dist = self.model(ctx_t, horizon=1)["dist"]   # torch.distributions
#
#         mu   = dist.mean.squeeze().item()
#         sigma= dist.stddev.squeeze().item() + 1e-9        # avoid div‑by‑0
#         return abs(value - mu) / sigma


# --------------- 2‑G  PatchTST, TTM‑R2, TEMPO, TabPFN‑TS -------------
# (identical pattern – omitted for brevity, add if you need all models)
# ---------------------------------------------------------------------


DETECTORS = {
    "RCF"     : lambda: RCFDetector(threshold=0.5),            # grade → use 0
    "Toto"    : lambda: TotoDetector(threshold=3.0),
    "Chronos" : lambda: ChronosDetector(threshold=3.0),
    #"Moirai"  : lambda: MoiraiDetector(),
}



# ---------------------------------------------------------------------
# 3.  Evaluation helpers (precision/recall & latency)
# ---------------------------------------------------------------------

WINDOW_MINUTES = 5
WIN = pd.Timedelta(minutes=WINDOW_MINUTES)

def event_metrics(pred_ts: List[pd.Timestamp],
                  label_ts: List[pd.Timestamp],
                  win=WIN):
    """Return (prec, rec, tp, fp, fn) at *event* granularity."""
    matched_lbl = set()
    matched_pred = set()

    for i, lbl in enumerate(label_ts):
        for j, p in enumerate(pred_ts):
            if abs(p - lbl) <= win:
                matched_lbl.add(i)
                matched_pred.add(j)

    tp = len(matched_lbl)
    fp = len(pred_ts) - len(matched_pred)
    fn = len(label_ts) - tp

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    return prec, rec, tp, fp, fn


# ---------------------------------------------------------------------
# 4.  Main benchmarking loop
# ---------------------------------------------------------------------
from collections import defaultdict

def evaluate():
    # ❶ correct defaultdict call
    results = defaultdict(
        lambda: {"tp":0, "fp":0, "fn":0,
                 "lat_sum":0.0, "cnt":0,
                 "params":0, "mem":0}
    )
    detail_rows = []

    streams = list(load_nab_streams())

    for model_name, init_fn in DETECTORS.items():
        # ❷ probe once for param‑count / mem
        probe = init_fn()
        if torch.cuda.is_available():
            # torch.cuda.reset_peak_memory_stats() clears the counter once per detector,
            # so max_memory_allocated() now reports only usage caused by that detector and the streams it sees.
            torch.cuda.reset_peak_memory_stats()

        if   hasattr(probe, "model"): m = probe.model
        elif hasattr(probe, "mod")  : m = probe.mod
        elif hasattr(probe, "fcst") : m = probe.fcst
        else                        : m = None
        results[model_name]["params"] = (
            sum(p.numel() for p in m.parameters()) / 1e6 if m else 0
        )
        del probe                     # free GPU/CPU memory

        # ── fresh detector for **every** CSV ──
        for stream_name, df, label_ts in streams:
            print("csv {}".format(stream_name))
            det = init_fn()           # new, clean forest
            stream_lat_sum = stream_cnt = 0
            scores, ts_list = [], []

            pred_ts = []
            for ts, val in zip(df["timestamp"], df["value"]):
                t0 = time.perf_counter()
                grade, hit_ts = det.update(ts, float(val))
                scores.append(grade)  # keep if you still need it
                if hit_ts is not None:  # collect *event* timestamps
                    pred_ts.append(hit_ts)
                dt = time.perf_counter() - t0

                results[model_name]["lat_sum"] += dt
                results[model_name]["cnt"]     += 1
                stream_lat_sum                += dt
                stream_cnt                    += 1

            # ②event‑level precision / recall
            p, r, tp, fp, fn = event_metrics(pred_ts, label_ts)

            # ③accumulate *counts* (not approximations)
            results[model_name]["tp"] += tp
            results[model_name]["fp"] += fp
            results[model_name]["fn"] += fn

            avg_ms = 1e3 * stream_lat_sum / stream_cnt
            detail_rows.append((model_name, stream_name, p, r, avg_ms))

            if torch.cuda.is_available():
                mem_now = torch.cuda.max_memory_allocated() / (1024 ** 3)
            else:
                # On CPU-only machines, we fall back to the resident-set size (psutil) and keep the maximum across streams.
                rss = psutil.Process(os.getpid()).memory_info().rss
                mem_now = rss / (1024 ** 3)
            results[model_name]["mem"] = max(results[model_name]["mem"], mem_now)
            gc.collect(); torch.cuda.empty_cache()

    summary_rows = []
    for n, d in results.items():
        prec = d["tp"] / (d["tp"] + d["fp"] + 1e-9)
        rec = d["tp"] / (d["tp"] + d["fn"] + 1e-9)
        lat = 1e3 * d["lat_sum"] / d["cnt"]  # ms/pt
        summary_rows.append((n, prec, rec, lat, d["params"], d["mem"]))

    summary_df = pd.DataFrame(summary_rows,
                              columns=["Model", "Precision", "Recall",
                                       "Avg ms/pt", "Params (M)", "Peak GB"])

    detail_df = pd.DataFrame(detail_rows,
                             columns=["Model", "Dataset", "Precision",
                                      "Recall", "Avg ms/pt"])

    return summary_df, detail_df

if __name__ == "__main__":
    # evaluate() → (summary_df, detail_df)
    summary_df, detail_df = evaluate()

    print("=== overall summary ===")
    print(summary_df.sort_values("Precision", ascending=False).to_string(index=False))

    print("\n=== per‑dataset details ===")
    # e.g. sort by model then dataset for readability
    print(detail_df.sort_values(["Model", "Dataset"]).to_string(index=False))

