
import csv
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "eda_results"
CHART_DIR = RESULTS_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True)


def load_csv(filename):
    path = RESULTS_DIR / filename

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bar_chart(filename, category, title, output, top=10):
    rows = load_csv(filename)[:top]

    labels = [r[category] for r in rows]
    counts = [int(r["count"]) for r in rows]

    plt.figure(figsize=(11, 6))
    plt.barh(labels[::-1], counts[::-1])
    plt.title(title)
    plt.xlabel("Number of requests")
    plt.tight_layout()
    plt.savefig(CHART_DIR / output, dpi=150)
    plt.close()


def monthly_chart(filename, title, output):
    rows = load_csv(filename)

    months = [r["month"] for r in rows]
    counts = [int(r["count"]) for r in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(months, counts, marker="o")
    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel("Number of requests")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CHART_DIR / output, dpi=150)
    plt.close()


for city, type_col, zip_col in [
    ("nyc", "complaint_type", "incident_zip"),
    ("la", "requesttype", "zipcode")
]:
    bar_chart(
        f"{city}_request_types.csv",
        type_col,
        f"{city.upper()} - Top 10 Request Types",
        f"{city}_request_types.png"
    )

    bar_chart(
        f"{city}_statuses.csv",
        "status",
        f"{city.upper()} - Request Status",
        f"{city}_statuses.png"
    )

    bar_chart(
        f"{city}_channels.csv",
        "open_data_channel_type"
        if city == "nyc" else "requestsource",
        f"{city.upper()} - Submission Channels",
        f"{city}_channels.png",
        top=15
    )

    bar_chart(
        f"{city}_postal_codes.csv",
        zip_col,
        f"{city.upper()} - Top 10 Postal Codes",
        f"{city}_postal_codes.png"
    )

    monthly_chart(
        f"{city}_monthly.csv",
        f"{city.upper()} - Monthly Request Volume",
        f"{city}_monthly.png"
    )

print("EDA charts saved in:", CHART_DIR)
