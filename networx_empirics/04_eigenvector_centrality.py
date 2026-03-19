"""
04 - Eigenvector Centrality and PageRank in Financial Networks
==============================================================
PhD Finance | Network Analysis

Eigenvector centrality: a node is important if it is connected
to other important nodes. This recursive definition is solved
via the dominant eigenvector of the adjacency matrix.

In finance: "too-connected-to-fail" institutions.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import eigh

# %% 1. Build a Fund-Stock Ownership Network (Bipartite → Projected)
# Real-world analog: 13F filings, mutual fund portfolios
# Projection onto stocks: two stocks are linked if co-held by same fund

funds = {
    "Vanguard":   ["AAPL", "MSFT", "GOOG", "JPM", "JNJ", "XOM", "NVDA", "BAC"],
    "BlackRock":  ["AAPL", "MSFT", "GOOG", "JPM", "UNH", "XOM", "META", "GS"],
    "Fidelity":   ["AAPL", "MSFT", "NVDA", "META", "JPM", "BAC", "C", "MS"],
    "StateStreet": ["AAPL", "MSFT", "GOOG", "JPM", "JNJ", "PFE", "MRK", "GS"],
    "Capital":    ["GOOG", "META", "NVDA", "UNH", "JNJ", "PFE", "XOM", "CVX"],
    "Wellington": ["JPM", "GS", "BAC", "C", "MS", "UNH", "MRK", "ABT"],
    "TRowe":      ["AAPL", "NVDA", "META", "XOM", "CVX", "COP", "SLB", "EOG"],
    "Bridgewater": ["JPM", "GS", "XOM", "CVX", "GOOG", "MSFT", "JNJ", "UNH"],
}

# Build bipartite graph
B = nx.Graph()
for fund, holdings in funds.items():
    B.add_node(fund, bipartite=0)
    for stock in holdings:
        B.add_node(stock, bipartite=1)
        B.add_edge(fund, stock)

# Project onto stock space: stocks linked by co-ownership
stocks = {n for n, d in B.nodes(data=True) if d["bipartite"] == 1}
G = nx.bipartite.weighted_projected_graph(B, stocks)

print("=" * 70)
print("INSTITUTIONAL CO-OWNERSHIP NETWORK — EIGENVECTOR CENTRALITY")
print("=" * 70)
print(f"Stocks: {G.number_of_nodes()}")
print(f"Co-ownership links: {G.number_of_edges()}")
print(f"Edge weight = number of funds co-holding both stocks")

# %% 2. Eigenvector Centrality — Mathematical Foundation
print("\n" + "-" * 70)
print("MATHEMATICAL FOUNDATION")
print("-" * 70)
print("""
  Eigenvector centrality x_i of node i satisfies:

      A·x = λ₁·x

  where A = adjacency matrix, λ₁ = largest eigenvalue (spectral radius)

      x_i = (1/λ₁) Σⱼ A_ij · x_j

  Meaning: node i's centrality = sum of neighbors' centralities,
  scaled by the spectral radius.

  By Perron-Frobenius theorem: for connected graphs with non-negative
  weights, the dominant eigenvector has ALL POSITIVE entries.

  FINANCIAL MEANING:
  A stock has high eigenvector centrality when it is co-held with
  other stocks that are THEMSELVES widely co-held → captures the
  "inner circle" of institutional portfolios.
""")

# %% 3. Compute eigenvector centrality — two methods
# Method 1: NetworkX (power iteration)
eigen_c = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)

# Method 2: Direct eigendecomposition (to see the math)
nodes = sorted(G.nodes())
A = nx.adjacency_matrix(G, nodelist=nodes, weight="weight").toarray().astype(float)
eigenvalues, eigenvectors = eigh(A)

# Dominant eigenvector (largest eigenvalue = last one from eigh)
lambda_1 = eigenvalues[-1]
v_1 = np.abs(eigenvectors[:, -1])  # Perron-Frobenius: all positive
v_1 = v_1 / v_1.sum()  # normalize

print(f"Spectral radius λ₁ = {lambda_1:.4f}")
print(f"  (This bounds the rate of shock propagation in the network)")

# Compare both methods
print("\n--- Eigenvector Centrality Rankings ---")
df = pd.DataFrame({
    "NX_Eigen": [eigen_c[n] for n in nodes],
    "Manual_Eigen": v_1,
    "Co-holders": [G.degree(n, weight="weight") for n in nodes],
}, index=nodes)
df = df.sort_values("NX_Eigen", ascending=False)
print(df.round(4).to_string())

# %% 4. PageRank — Eigenvector centrality with damping
# PageRank adds a "teleportation" probability (damping factor d)
# to handle dangling nodes and improve convergence
# In finance: captures "random investor" behavior

pagerank = nx.pagerank(G, weight="weight", alpha=0.85)

print("\n--- PageRank vs Eigenvector Centrality ---")
df["PageRank"] = [pagerank[n] for n in df.index]

print(f"\n{'Stock':>6s} {'Eigenvector':>12s} {'PageRank':>10s} {'Rank_E':>7s} {'Rank_P':>7s}")
print("-" * 50)
rank_e = df["NX_Eigen"].rank(ascending=False).astype(int)
rank_p = df["PageRank"].rank(ascending=False).astype(int)
for stock in df.index:
    print(f"{stock:>6s} {df.loc[stock, 'NX_Eigen']:12.4f} {df.loc[stock, 'PageRank']:10.4f} "
          f"{rank_e[stock]:7d} {rank_p[stock]:7d}")

# %% 5. Katz Centrality — with attenuation factor
# Katz: counts ALL paths, not just shortest, with exponential decay
# In finance: captures indirect contagion channels
alpha_katz = 0.8 / lambda_1  # must be < 1/λ₁ for convergence
katz_c = nx.katz_centrality(G, alpha=alpha_katz, weight="weight")

print(f"\n--- Katz Centrality (α = {alpha_katz:.4f} < 1/λ₁ = {1/lambda_1:.4f}) ---")
print("  Katz counts ALL walks of all lengths, decayed by α^k")
print("  Finance: captures cascading/indirect contagion channels")
top5_katz = sorted(katz_c.items(), key=lambda x: x[1], reverse=True)[:5]
for stock, val in top5_katz:
    print(f"    {stock:6s}: {val:.4f}")

# %% 6. Visualization
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
pos = nx.spring_layout(G, seed=42, k=1.8)

# Eigenvector centrality
ax = axes[0]
vals = np.array([eigen_c[n] for n in G.nodes()])
sizes = 300 + 2500 * (vals - vals.min()) / (vals.max() - vals.min())
nx.draw(G, pos, ax=ax, with_labels=True, node_size=sizes,
        node_color=vals, cmap="YlOrRd", font_size=9, font_weight="bold",
        edge_color="gray", alpha=0.8, width=0.5)
ax.set_title("Eigenvector Centrality\n(connected to important nodes)", fontsize=12)

# PageRank
ax = axes[1]
vals = np.array([pagerank[n] for n in G.nodes()])
sizes = 300 + 2500 * (vals - vals.min()) / (vals.max() - vals.min())
nx.draw(G, pos, ax=ax, with_labels=True, node_size=sizes,
        node_color=vals, cmap="YlGnBu", font_size=9, font_weight="bold",
        edge_color="gray", alpha=0.8, width=0.5)
ax.set_title("PageRank\n(with random investor teleportation)", fontsize=12)

# Bipartite original
ax = axes[2]
fund_nodes = [n for n in B.nodes() if B.nodes[n].get("bipartite") == 0]
stock_nodes = [n for n in B.nodes() if B.nodes[n].get("bipartite") == 1]
pos_bip = {}
for i, f in enumerate(fund_nodes):
    pos_bip[f] = (0, i * 1.2)
for i, s in enumerate(sorted(stock_nodes)):
    pos_bip[s] = (2, i * 0.65)
nx.draw_networkx_nodes(B, pos_bip, nodelist=fund_nodes, node_color="#f59e0b",
                       node_size=800, ax=ax)
nx.draw_networkx_nodes(B, pos_bip, nodelist=stock_nodes, node_color="#3b82f6",
                       node_size=500, ax=ax)
nx.draw_networkx_labels(B, pos_bip, font_size=7, ax=ax)
nx.draw_networkx_edges(B, pos_bip, alpha=0.3, ax=ax)
ax.set_title("Bipartite Fund-Stock Network\n(projected to create co-ownership graph)", fontsize=12)

plt.tight_layout()
plt.savefig("/mnt/nas/Class/2026/PHD/networkX/04_eigenvector_pagerank.png", dpi=150)
plt.show()
print("\nFigure saved: 04_eigenvector_pagerank.png")

# %% 7. Sensitivity Analysis: What if a big fund exits?
print("\n--- Shock Simulation: Fund Liquidation ---")
print("  What happens to eigenvector centrality when a fund exits?\n")

baseline_eigen = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)

for fund in ["Vanguard", "BlackRock", "Bridgewater"]:
    B_shock = B.copy()
    # Remove all edges from this fund
    edges_to_remove = list(B_shock.edges(fund))
    B_shock.remove_edges_from(edges_to_remove)
    B_shock.remove_node(fund)

    stocks_shock = {n for n, d in B_shock.nodes(data=True) if d.get("bipartite") == 1}
    G_shock = nx.bipartite.weighted_projected_graph(B_shock, stocks_shock)

    eigen_shock = nx.eigenvector_centrality(G_shock, weight="weight", max_iter=1000)

    print(f"  If {fund} liquidates all holdings:")
    for stock in sorted(funds[fund])[:5]:
        if stock in eigen_shock:
            delta = eigen_shock[stock] - baseline_eigen.get(stock, 0)
            print(f"    {stock}: {baseline_eigen.get(stock, 0):.4f} → {eigen_shock[stock]:.4f} "
                  f"(Δ = {delta:+.4f})")
    print()

# %% 8. Exercises
print("=" * 70)
print("EXERCISES")
print("=" * 70)
print("""
1. Verify Perron-Frobenius: show that the dominant eigenvector of A
   has all positive entries. What happens for a disconnected graph?

2. Vary the PageRank damping factor α from 0.5 to 0.99.
   Plot how rankings change. At α→1, PageRank → eigenvector centrality.
   Why does Google use α=0.85?

3. Build the ownership network from actual 13F data (SEC EDGAR).
   Which stocks have the highest eigenvector centrality?
   Does it correlate with market cap?

4. Compute the HITS algorithm: nx.hits(G)
   Hubs = stocks held by many funds; Authorities = stocks that
   "define" a fund's strategy. How do hub/authority scores differ
   from eigenvector centrality?

5. (Research) Read Greenwood & Thesmar (2011) "Stock price fragility"
   — they use portfolio overlap to measure fire-sale vulnerability.
   How does their measure relate to eigenvector centrality in the
   co-ownership network?
""")
