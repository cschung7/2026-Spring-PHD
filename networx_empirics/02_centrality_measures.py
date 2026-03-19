"""
02 - Centrality Measures in Financial Networks
===============================================
PhD Finance | Network Analysis

Key Question: Which institution is "most important" in the financial system?
Different centrality measures capture different notions of importance.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% 1. Build a Stock Correlation Network
# Nodes = stocks, edges = significant correlations between returns
# This is a common construction in financial network analysis
# (Mantegna 1999, Tumminello et al. 2005)

np.random.seed(42)

sectors = {
    "Tech":    ["AAPL", "MSFT", "GOOG", "NVDA", "META"],
    "Finance": ["JPM", "GS", "BAC", "C", "MS"],
    "Energy":  ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Health":  ["JNJ", "PFE", "UNH", "MRK", "ABT"],
}

G = nx.Graph()
for sector, tickers in sectors.items():
    for t in tickers:
        G.add_node(t, sector=sector)

# Intra-sector correlations (strong)
for sector, tickers in sectors.items():
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            corr = np.random.uniform(0.5, 0.9)
            G.add_edge(tickers[i], tickers[j], weight=corr)

# Inter-sector correlations (weaker, sparser)
cross_links = [
    ("AAPL", "JPM", 0.35), ("MSFT", "GS", 0.30),
    ("GOOG", "UNH", 0.25), ("NVDA", "XOM", 0.20),
    ("JPM", "XOM", 0.40),  ("GS", "CVX", 0.38),
    ("BAC", "JNJ", 0.22),  ("C", "PFE", 0.28),
    ("META", "MRK", 0.18), ("MS", "COP", 0.33),
    ("JPM", "UNH", 0.31),  ("XOM", "JNJ", 0.26),
]
for u, v, w in cross_links:
    G.add_edge(u, v, weight=w)

print("=" * 70)
print("STOCK CORRELATION NETWORK — CENTRALITY ANALYSIS")
print("=" * 70)
print(f"Stocks: {G.number_of_nodes()}, Correlation links: {G.number_of_edges()}")

# %% 2. Four Centrality Measures
degree_c = nx.degree_centrality(G)
between_c = nx.betweenness_centrality(G, weight="weight")
close_c = nx.closeness_centrality(G)
eigen_c = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)

# %% 3. Compare centralities
df = pd.DataFrame({
    "Sector": [G.nodes[n]["sector"] for n in G.nodes()],
    "Degree": [degree_c[n] for n in G.nodes()],
    "Betweenness": [between_c[n] for n in G.nodes()],
    "Closeness": [close_c[n] for n in G.nodes()],
    "Eigenvector": [eigen_c[n] for n in G.nodes()],
}, index=list(G.nodes()))

print("\n--- Centrality Rankings ---")
for measure in ["Degree", "Betweenness", "Closeness", "Eigenvector"]:
    top3 = df[measure].nlargest(5)
    print(f"\n  {measure} Centrality (Top 5):")
    for ticker, val in top3.items():
        print(f"    {ticker:6s} ({df.loc[ticker, 'Sector']:8s}): {val:.4f}")

# %% 4. Interpretation for Finance
print("\n" + "=" * 70)
print("FINANCIAL INTERPRETATION OF CENTRALITY")
print("=" * 70)
print("""
  Degree Centrality
    = fraction of stocks a given stock is correlated with
    → Measures DIVERSIFICATION DIFFICULTY
    → High degree = hard to diversify away from this stock

  Betweenness Centrality
    = fraction of shortest paths passing through a stock
    → Measures INFORMATION BROKERAGE / CONTAGION BOTTLENECK
    → High betweenness = removing this stock fragments the network
    → Key for identifying SYSTEMIC RISK intermediaries

  Closeness Centrality
    = inverse average distance to all other stocks
    → Measures how quickly shocks PROPAGATE from this stock
    → High closeness = shock reaches the whole market fastest

  Eigenvector Centrality
    = importance based on connections to OTHER important nodes
    → Measures RECURSIVE INFLUENCE (Google's PageRank ancestor)
    → High eigenvector = connected to the "inner circle" of the market
    → Captures the "too-connected-to-fail" phenomenon
""")

# %% 5. Correlation matrix of centrality measures
corr = df[["Degree", "Betweenness", "Closeness", "Eigenvector"]].corr()
print("--- Correlation Between Centrality Measures ---")
print(corr.round(3).to_string())
print("\n  (High correlation → measures capture similar notion of importance)")
print("  (Low correlation → measures reveal different structural roles)")

# %% 6. Visualization — four centrality views
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
measures = ["Degree", "Betweenness", "Closeness", "Eigenvector"]
cmaps = ["Blues", "Oranges", "Greens", "Reds"]

sector_colors = {"Tech": "#3b82f6", "Finance": "#ef4444",
                 "Energy": "#f59e0b", "Health": "#10b981"}
pos = nx.spring_layout(G, seed=42, k=1.5)

for ax, measure, cmap in zip(axes.flat, measures, cmaps):
    values = df[measure].values
    node_sizes = 300 + 3000 * (values - values.min()) / (values.max() - values.min())
    node_colors = [sector_colors[G.nodes[n]["sector"]] for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, alpha=0.8, edgecolors="black")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold")

    top_node = df[measure].idxmax()
    ax.set_title(f"{measure} Centrality\n(Top: {top_node})", fontsize=13)

    # Legend
    for sector, color in sector_colors.items():
        ax.scatter([], [], c=color, s=100, label=sector)
    ax.legend(loc="lower left", fontsize=8)

plt.suptitle("Stock Correlation Network — Centrality Comparison", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("/mnt/nas/Class/2026/PHD/networkX/02_centrality_comparison.png", dpi=150)
plt.show()
print("\nFigure saved: 02_centrality_comparison.png")

# %% 7. Rank correlation — do measures agree?
from scipy.stats import spearmanr

print("\n--- Spearman Rank Correlations ---")
for i in range(len(measures)):
    for j in range(i + 1, len(measures)):
        rho, pval = spearmanr(df[measures[i]], df[measures[j]])
        print(f"  {measures[i]:12s} vs {measures[j]:12s}: ρ = {rho:+.3f}  (p = {pval:.4f})")

# %% 8. Exercises
print("\n" + "=" * 70)
print("EXERCISES")
print("=" * 70)
print("""
1. Remove all inter-sector edges. Recalculate betweenness.
   What happens and why? (Hint: think about bridge nodes)

2. Compute PageRank: nx.pagerank(G, weight='weight')
   Compare with eigenvector centrality. When do they diverge?

3. Add a new "Central Bank" node connected to all Finance stocks
   with weight=0.95. How does this change eigenvector centrality
   for non-finance stocks?

4. Threshold the network: keep only edges with corr > 0.4.
   Which centrality measure changes the MOST? Why?

5. (Research) Read Billio et al. (2012) "Econometric measures of
   connectedness and systemic risk in the finance and insurance sectors"
   — how do they use Granger-causality networks with centrality?
""")
