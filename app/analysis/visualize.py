import matplotlib
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt


# def plot_sentiment(sentiment: dict, out_path: str):
#     labels = sentiment["counts"].keys()
#     values = sentiment["counts"].values()

#     plt.figure(figsize=(5, 5))
#     plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
#     plt.title("Parent Feedback Sentiment")
#     plt.savefig(out_path, bbox_inches="tight")
#     plt.close()

def plot_sentiment(sentiment: dict, out_path: str):
    if not sentiment or "counts" not in sentiment:
        return

    counts = sentiment.get("counts", {})

    if not counts:
        return

    labels = []
    values = []

    for k, v in counts.items():
        if v is not None and v > 0:
            labels.append(k)
            values.append(v)

    # 🚨 FINAL GUARD: nothing valid to plot
    if not values or sum(values) == 0:
        return

    plt.figure(figsize=(5, 5))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("Parent Feedback Sentiment")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_keywords(keywords: list[str], out_path: str):
    if not keywords:
        return
    plt.figure(figsize=(8, 4))
    plt.barh(keywords[::-1], range(len(keywords)))
    plt.title("Top Feedback Keywords")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_topics(topics: list[dict], out_path: str):
    labels = [f"Topic {t['topic_id']}" for t in topics]
    sizes = [len(t["keywords"]) for t in topics]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, sizes)
    plt.title("Detected Feedback Topics")
    plt.ylabel("Keyword Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
