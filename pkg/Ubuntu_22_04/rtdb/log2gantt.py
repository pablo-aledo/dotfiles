import json
import sys
import os
import plotly.express as px
import pandas as pd
import datetime

def nanoseconds_to_standard_time(nanoseconds):
    seconds = nanoseconds / 1e9  # Convert nanoseconds to seconds
    return datetime.datetime.utcfromtimestamp(seconds).strftime('%Y-%m-%d %H:%M:%S.%f')

def process_log(file_path):

    list_message_start_end = [
        ("message", "message end")
    ]

    final_events = []

    for message_start, message_end in list_message_start_end:
        starting_events = {}

        with open(file_path, 'r') as file:
            for line in file:
                jsonline = json.loads(line)
                if "message" in jsonline and jsonline["message"] == message_start:
                    key = jsonline.get("key", message_start)
                    starting_events[key] = nanoseconds_to_standard_time(jsonline["time"])
                if "message" in jsonline and jsonline["message"] == message_end:
                    key = jsonline.get("key", message_start)
                    category = jsonline.get("category", message_start)
                    final_events.append({
                        "event": message_start,
                        "key": key,
                        "category": category,
                        "start": starting_events[key],
                        "end": nanoseconds_to_standard_time(jsonline["time"])
                    })

    return final_events


if __name__ == "__main__":
    log_file = sys.argv[1]

    final_events = process_log(log_file)

    for event in final_events:
        print(event)

    df = pd.DataFrame([ dict(
        Task=event["key"],
        Start=event["start"],
        Finish=event["end"],
        Category=event["category"]
    ) for event in final_events ])

    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Category"
    )

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1s",
                         step="second",
                         stepmode="backward"),
                    dict(count=1,
                         label="1m",
                         step="minute",
                         stepmode="backward"),
                    dict(count=5,
                         label="5m",
                         step="minute",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    fig.update_traces(marker=dict(line=dict(width=1, color="black")))

    fig.show()
