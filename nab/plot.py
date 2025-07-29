import plotly.graph_objs as go
import pandas as pd
from datetime import datetime
import pytz


def plot2(df, xcol, ycol, label_ts, anomaly_ts, ylabel, ts_format="%Y-%m-%d @ %H:%M"):
    filtered_df = df.copy()
    filtered_df[xcol] = pd.to_datetime(filtered_df[xcol])

    # Convert anomaly timestamps
    anomaly_timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in anomaly_ts]

    # Create the main line plot
    fig = go.Figure([go.Scatter(x=filtered_df[xcol], y=filtered_df[ycol], mode='lines', name='Raw Data')])

    # Add detected anomalies as red 'x' markers
    fig.add_trace(go.Scatter(x=anomaly_timestamps,
                             y=[filtered_df[filtered_df[xcol] == ts][ycol].values[0] for ts in anomaly_timestamps if
                                not filtered_df[filtered_df[xcol] == ts].empty],
                             mode='markers', marker=dict(color='red', symbol='x', size=10), name='Detected Anomaly'))

    # Shade areas and add tooltips for label_ts
    for period in label_ts:
        print(period)
        # split will return a list containing the original string if the delimiter is not found
        start = None
        end = None
        if ' - ' in period:
            start_str, end_str = period.split(' - ')
            start = datetime.strptime(start_str, ts_format)
            end = datetime.strptime(end_str, ts_format)
        else:
            start = datetime.strptime(period, ts_format)
            end = start
        fig.add_vrect(x0=start, x1=end, fillcolor="yellow", opacity=0.2, line_width=0, annotation_text='Label',
                      annotation_position='top left', annotation=dict(font_size=10))
        # Add labeled anomalies as larger blue circle markers, if they exist in the data
        if not filtered_df[filtered_df[xcol] == start].empty:
            fig.add_trace(go.Scatter(x=[start],
                                     y=[filtered_df[filtered_df[xcol] == start][ycol].values[0]],
                                     mode='markers', marker=dict(color='green', symbol='circle', size=12, opacity=0.5),
                                     name='Labeled Anomaly'))

    # Update layout
    fig.update_layout(title=f'{ylabel} Over Time',
                      xaxis_title=xcol,
                      yaxis_title=ylabel,
                      hovermode='x unified')

    fig.show()


import pandas as pd
from datetime import datetime
import plotly.graph_objects as go


def plot3(df, xcol, ycol, label_ts, anomaly_ts, ylabel, ts_format="%Y-%m-%d @ %H:%M"):
    filtered_df = df.copy()
    filtered_df[xcol] = pd.to_datetime(filtered_df[xcol])

    # Convert anomaly timestamps
    anomaly_timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in anomaly_ts]

    # Create the main line plot
    fig = go.Figure([go.Scatter(
        x=filtered_df[xcol],
        y=filtered_df[ycol],
        mode='lines',
        name='Raw Data',
        line=dict(color='blue')
    )])

    # Plot detected anomalies (red 'x')
    detected_y = []
    for ts in anomaly_timestamps:
        matched = filtered_df[filtered_df[xcol] == ts]
        if not matched.empty:
            detected_y.append(matched[ycol].values[0])

    fig.add_trace(go.Scatter(
        x=anomaly_timestamps,
        y=detected_y,
        mode='markers',
        marker=dict(color='red', symbol='x', size=10),
        name='Detected Anomaly'
    ))

    # Add labeled anomalies with vertical and horizontal offset
    # Adjust these offsets as needed
    vertical_offset = 0.01  # proportionally small offset
    horizontal_offset = pd.Timedelta(milliseconds=50)  # slight shift in time

    for period in label_ts:
        if ' - ' in period:
            start_str, end_str = period.split(' - ')
            start = datetime.strptime(start_str, ts_format)
            end = datetime.strptime(end_str, ts_format)
        else:
            start = datetime.strptime(period, ts_format)
            end = start

        # Add vertical rectangle with annotation
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="yellow",
            opacity=0.2,
            line_width=0,
            annotation_text='Label',
            annotation_position='top left',
            annotation=dict(font_size=10)
        )

        # Plot labeled anomaly if data point exists at 'start'
        matched = filtered_df[filtered_df[xcol] == start]
        if not matched.empty:
            original_value = matched[ycol].values[0]

            # Apply vertical and horizontal offsets
            labeled_time = start + horizontal_offset
            # Scale vertical offset by data value if needed, or keep constant
            adjusted_value = original_value + max(vertical_offset * original_value, vertical_offset)

            fig.add_trace(go.Scatter(
                x=[labeled_time],
                y=[adjusted_value],
                mode='markers',
                # Use a distinct marker style to stand out
                marker=dict(color='green', symbol='star-open', size=20, line=dict(color='black', width=2)),
                name='Labeled Anomaly'
            ))

    fig.update_layout(
        title=f'{ylabel} Over Time',
        xaxis_title=xcol,
        yaxis_title=ylabel,
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    fig.show()
