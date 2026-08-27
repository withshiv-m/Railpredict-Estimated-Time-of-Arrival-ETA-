import pandas as pd
import os


# -------------------------------
# Load Dataset
# -------------------------------

file_path = os.path.join(
    os.path.dirname(__file__),
    "train_data.csv"
)

df = pd.read_csv(file_path)


# -------------------------------
# Basic Dataset Information
# -------------------------------

print("\n==============================")
print("       DATASET INFO")
print("==============================")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())


# -------------------------------
# Basic Statistics
# -------------------------------

print("\n==============================")
print("       BASIC STATISTICS")
print("==============================")

print("\nAverage Speed:")
print(round(df["speed"].mean(), 2), "km/h")

print("\nAverage Current Delay:")
print(round(df["current_delay"].mean(), 2), "minutes")

print("\nAverage Rain:")
print(round(df["rain"].mean(), 2))

print("\nAverage Travel Time:")
print(round(df["travel_time"].mean(), 2), "minutes")


# -------------------------------
# Baseline ETA Function
# -------------------------------

def calculate_baseline_eta(row):

    distance = row["distance_remaining"]
    speed = row["speed"]
    current_delay = row["current_delay"]

    if speed <= 0:
        return None

    # Basic travel time
    travel_time = (distance / speed) * 60

    # Add current delay
    eta = travel_time + current_delay

    return eta


# -------------------------------
# Single Train Prediction
# -------------------------------

train_id = input("\nEnter Train ID: ")

train = df[
    df["train_id"].astype(str).str.strip().str.lower()
    == train_id.strip().lower()
]


if train.empty:

    print("\nTrain not found!")

else:

    row = train.iloc[0]

    baseline_eta = calculate_baseline_eta(row)

    actual_time = row["travel_time"]

    error = abs(actual_time - baseline_eta)

    print("\n==============================")
    print("        ETA FORECAST")
    print("==============================")

    print("Train:", row["train_id"])

    print(
        "Route:",
        row["source"],
        "→",
        row["destination"]
    )

    print(
        "Current Station:",
        row["current_station"]
    )

    print(
        "Next Station:",
        row["next_station"]
    )

    print(
        "Distance:",
        row["distance_remaining"],
        "km"
    )

    print(
        "Speed:",
        row["speed"],
        "km/h"
    )

    print(
        "Current Delay:",
        row["current_delay"],
        "minutes"
    )

    print(
        "Baseline ETA:",
        round(baseline_eta, 2),
        "minutes"
    )

    print(
        "Actual Travel Time:",
        actual_time,
        "minutes"
    )

    print(
        "Prediction Error:",
        round(error, 2),
        "minutes"
    )

    print("==============================")


# -------------------------------
# Evaluate All 39 Trains
# -------------------------------

df["baseline_eta"] = df.apply(
    calculate_baseline_eta,
    axis=1
)

df["prediction_error"] = (
    df["travel_time"] - df["baseline_eta"]
).abs()


print("\n===================================")
print("       BASELINE MODEL RESULTS")
print("===================================")

print(
    "Average Baseline ETA:",
    round(df["baseline_eta"].mean(), 2),
    "minutes"
)

print(
    "Average Actual Travel Time:",
    round(df["travel_time"].mean(), 2),
    "minutes"
)

print(
    "Average Prediction Error:",
    round(df["prediction_error"].mean(), 2),
    "minutes"
)

print(
    "Minimum Error:",
    round(df["prediction_error"].min(), 2),
    "minutes"
)

print(
    "Maximum Error:",
    round(df["prediction_error"].max(), 2),
    "minutes"
)

print("===================================")