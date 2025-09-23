import matplotlib.pyplot as plt


methods = ["SPLADE Hybrid", "RRF Fusion", "Neural Retrieval"]
precision = [0.667, 0.25, 0.34]
recall = [0.333, 0.125, 0.7083]
f1 = [0.444, 0.166, 0.459]

x = range(len(methods))

plt.figure(figsize=(10,6))
plt.bar(x, precision, width=0.2, label="Precision")
plt.bar([i+0.2 for i in x], recall, width=0.2, label="Recall")
plt.bar([i+0.4 for i in x], f1, width=0.2, label="F1-Score")

plt.xticks([i+0.2 for i in x], methods, rotation=20)
plt.ylabel("Score")
plt.title("Performance Comparison of Retrieval Methods")
plt.legend()
plt.show()


"""
1. Precision, Recall, and F1 Comparison

A bar chart showing precision, recall, and F1 for each experiment (e.g., SPLADE baseline, RRF fusion, neural retrieval).

2. Cumulative Recall Curve

A line chart showing how recall improves as you increase the number of retrieved documents (top-5, top-10, top-20, top-50).

This shows whether your model finds relevant docs early or only after retrieving many.

 3. Confusion View: Retrieved vs. Missing Docs

A pie chart or stacked bar chart showing the proportion of:

Relevant retrieved

Irrelevant retrieved

Missing relevant documents

🔹 4. Per-Query Breakdown

If you evaluate on multiple queries later, you can plot per-query precision and recall in a grouped bar chart.

Helps show consistency of methods across different biomedical questions.

🔹 5. PR Curve or F1 vs. k

If you can vary the cutoff (top-k retrieved documents), you can plot a Precision-Recall curve or F1 vs. top-k chart to show model trade-offs."""