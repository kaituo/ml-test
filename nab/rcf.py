from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from lib.python3_8.site_packages.rcf.trcf_model import TRandomCutForestModel as TRCF
from plot import plot3
from collections import deque
from datetime import datetime, timedelta

probationPercent = 0

def plot(df, ylabel):
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['value'])
    plt.xlabel('Timestamp')
    plt.ylabel(ylabel)
    plt.title(ylabel + ' Over Time')
    plt.show()

def getProbationaryLength(numRows):
    return min(
      math.floor(probationPercent * numRows),
      probationPercent * 5000
    )

def perMinute(df, model, shingle_size, grade_threshold=0.0):
    count = 0
    anomaly_ts = []
    fixed_size_deque = deque(maxlen=shingle_size)

    num_rows = len(df)
    probationLength = getProbationaryLength(num_rows)
    probationTimestamp = None

    # Access the 'value' column
    for index, row in df.iterrows():
        # process needs an array of doubles
        floats = [row['value']]
        double_array = np.array(floats, dtype='float64')
        descriptor = model.process(double_array, 0)
        fixed_size_deque.append(row['timestamp'])
        if index >= probationLength and probationTimestamp is None:
            probationTimestamp = row['timestamp']
        if descriptor.getAnomalyGrade() > grade_threshold:
            expected = descriptor.getExpectedValuesList()[0][0] if descriptor.getExpectedValuesList() is not None else 0
            if descriptor.getRelativeIndex() < 0:
                if descriptor.getPastValues() is None:
                    continue
                actual = descriptor.getPastValues()[0]
            else:
                actual = row['value']
            #if actual > 1.5 * expected:
            print("timestamp {} grade {} value {} expected {} pastValue {} relativeIndex {} actual {}".format(row['timestamp'],
                                                                                                    descriptor.getAnomalyGrade(),
                                                                                                    row['value'],
                                                                                                    descriptor.getExpectedValuesList()[
                                                                                                        0] if descriptor.getExpectedValuesList() is not None else [],
                                                                                                    descriptor.getPastValues(),
                                                                                                    descriptor.getRelativeIndex(),
                                                                                                    floats)
                  )
            count += 1
            if descriptor.getRelativeIndex() == 0:
                anomaly_ts.append(row['timestamp'])
            else:
                # fixed_size_deque contains current timestamp. descriptor.getRelativeIndex() returning -1 means we are gonna get 2nd to last item
                anomaly_ts.append(fixed_size_deque[descriptor.getRelativeIndex() - 1])
    print("total anomalies: ", count)
    return anomaly_ts, probationTimestamp

def perDay(df, model, freq):
    # Convert the timestamp column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Set the timestamp column as the index
    df = df.set_index('timestamp')

    # Resample the data to daily frequency, taking the max value for each day
    daily_max = df.resample(freq).max()

    count = 0
    # Access the 'value' column
    for index, row in daily_max.iterrows():
        # process needs an array of doubles
        floats = [row['value']]
        double_array = np.array(floats, dtype='float64')
        descriptor = model.process(double_array)
        if descriptor.getAnomalyGrade() > 0:
            expected = descriptor.getExpectedValuesList()[0][0] if descriptor.getExpectedValuesList() is not None else 0
            if descriptor.getRelativeIndex() < 0:
                actual = descriptor.getPastValues()[0]
            else:
                actual = row['value']
            if actual > 1.5 * expected:
                print("timestamp {} grade {} value {} expected {} pastValue {} relativeIndex {}".format(index, descriptor.getAnomalyGrade(), row['value'], descriptor.getExpectedValuesList()[0] if descriptor.getExpectedValuesList() is not None else [], descriptor.getPastValues(), descriptor.getRelativeIndex()))
                count += 1
    print("total anomalies: ", count)

def run_experiment_parkservice(
        csv: str,
        shingle_size: int,
        num_trees: int,
        output_after: int,
        ploting: bool,
        ylabel: str,
        label: List[str],
        ts_format: str = "%Y-%m-%d @ %H:%M",
        sample_size: int=256,
        ignore_delta_threshold: float = 0.0,
        threadhold: float = 0.0
        ):
    """Run an experiment on a single Pandas dataframe.

    Use the input parameters to construct an RCF model. Perform streaming
    training and scoring on the data stored in the input dataframe.

    """
    print("Run experiment for " + csv)
    dimensions = 1

    # parenthesis for multi-line statements
    model = TRCF(rcf_dimensions=shingle_size*dimensions, shingle_size=shingle_size, num_trees=num_trees,
                 output_after=output_after, anomaly_rate=0.005, z_factor=3, score_differencing=0.5,
                 ignore_delta_threshold=ignore_delta_threshold, sample_size=sample_size, ignore_delta_threshold_ratio=0.2)
    df = pd.read_csv(csv)

    anomaly_ts, probationCutOffTimestamp = perMinute(df, model, shingle_size, threadhold)
    #perDay(df, model, 'D')
    #perDay(df, model, '30T')

    print(probationCutOffTimestamp)
    precision, recall = compute_precision_recall(label, anomaly_ts, probationTimestamp=probationCutOffTimestamp)
    print(f"csv {csv}: Precision: {precision:.2f}, Recall: {recall:.2f}")
    print(f"{anomaly_ts}")
    if ploting:
        #plot(df, ylabel)
        plot3(df, 'timestamp', 'value', label, anomaly_ts, ylabel, ts_format)

from datetime import datetime, timedelta

def compute_precision_recall(label_timestamps, anomaly_timestamps, time_window_minutes=5, probationTimestamp=None):
    # Convert timestamps to datetime objects
    labels = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in label_timestamps]
    anomalies = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in anomaly_timestamps]

    # Parse the probation timestamp if provided; otherwise, use a very old date to include all data
    if probationTimestamp is not None:
        probation_dt = datetime.strptime(probationTimestamp, "%Y-%m-%d %H:%M:%S")
    else:
        # If no probationTimestamp is provided, set to an old date so everything counts
        probation_dt = datetime.strptime("1971-02-16 07:10:00", "%Y-%m-%d %H:%M:%S")
    print(probation_dt)
    # Filter out labels and anomalies that occur before the probation period
    labels = [l for l in labels if l >= probation_dt]
    print(labels)
    anomalies = [a for a in anomalies if a >= probation_dt]

    # Define a time window around each label (e.g., ±5 minutes)
    time_window = timedelta(minutes=time_window_minutes)

    # Calculate true positives (TP), false positives (FP), and false negatives (FN)
    tp = 0
    fp = 0
    fn = 0

    for anomaly in anomalies:
        if any(label - time_window <= anomaly <= label + time_window for label in labels):
            tp += 1
        else:
            fp += 1

    for label in labels:
        if not any(label - time_window <= anomaly <= label + time_window for anomaly in anomalies):
            fn += 1

    # Compute precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall


if __name__ == "__main__":
    ts_format = "%Y-%m-%d %H:%M:%S"
    # default
    # run_experiment_parkservice('ec2_cpu_utilization_24ae8d.csv', 288, 50, 32, True, 'CPU Utilization', [
    #     "2014-02-26 22:05:00",
    #     "2014-02-27 17:15:00"
    # ], ts_format, threadhold=0.5)
    # run_experiment_parkservice('ec2_network_in_257a54.csv', 8, 50, 32, True, 'network in', [
    #     "2014-04-15 16:44:00"
    # ], ts_format, threadhold=0.5)
    # run_experiment_parkservice('ec2_disk_write_bytes_1ef3de.csv', 8, 50, 32, True, 'disk write', [
    #     "2014-03-10 21:09:00"
    # ], ts_format, threadhold=0.5)
    run_experiment_parkservice('rds_cpu_utilization_e47b3b.csv', 8, 50, 32, True, 'rds cpu', [
        "2014-04-13 06:52:00",
        "2014-04-18 23:27:00"
    ], ts_format, threadhold=0.5)
    # run_experiment_parkservice('rds_cpu_utilization_cc0c53.csv', 8, 50, 32, True, 'rds cpu cc0c53', [
    #     "2014-02-25 07:15:00",
    #     "2014-02-27 00:50:00"
    # ], ts_format, threadhold=0.5)

    # optimal, shingle, normalize, ignore threshold
    # run_experiment_parkservice('ec2_cpu_utilization_24ae8d.csv', 8, 30, 32, True, 'CPU Utilization', ignore_delta_threshold=1.5)
    # run_experiment_parkservice('ec2_network_in_257a54.csv', 8, 30, 32, True, 'network in', ignore_delta_threshold=5000000)
    # run_experiment_parkservice('ec2_disk_write_bytes_1ef3de.csv', 16, 30, 32, True, 'disk write', ignore_delta_threshold=400000000)#
    # run_experiment_parkservice('rds_cpu_utilization_e47b3b.csv', 16, 30, 32, True, 'rds cpu', ignore_delta_threshold=0)
    # run_experiment_parkservice('rds_cpu_utilization_cc0c53.csv', 8, 30, 32, True, 'rds cpu cc0c53', ignore_delta_threshold=15)

    # sample size
    # run_experiment_parkservice('ec2_cpu_utilization_24ae8d.csv', 8, 30, 32, False, 'CPU Utilization',
    #                            sample_size=1024)
    # run_experiment_parkservice('ec2_network_in_257a54.csv', 8, 30, 32, False, 'network in', sample_size=256)
    # run_experiment_parkservice('ec2_disk_write_bytes_1ef3de.csv', 16, 30, 32, False, 'disk write', sample_size=256)
    # run_experiment_parkservice('rds_cpu_utilization_e47b3b.csv', 16, 30, 32, False, 'rds cpu', sample_size=256)
    # run_experiment_parkservice('rds_cpu_utilization_cc0c53.csv', 8, 30, 32, False, 'rds cpu cc0c53', sample_size=256)

    # test an anomaly before the first 100 points and have late detection
    #run_experiment_parkservice('dataset.csv', 16, 30, 32, False, 'test past avlue', sample_size=256)
