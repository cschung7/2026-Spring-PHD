"""
05 - Zachary's Karate Club: Community Detection Applied to Finance
=================================================================
PhD Finance | Network Analysis

The Karate Club graph is the "hello world" of network science.
Here we use it as a template for community detection techniques
applied to financial networks: detecting market regimes,
sector clustering, and contagion boundaries.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from collections import defaultdict

# %% 1. Load the Karate Club Graph
K = nx.karate_club_graph()

# Ground truth: which faction each member belongs to
true_labels = {}
for node in K.nodes():
    true_labels[node] = K.nodes[node]["club"]  # 'Mr. Hi' or 'Officer'

print("=" * 70)
print("ZACHARY'S KARATE CLUB — COMMUNITY DETECTION")
print("=" * 70)
print(f"Members: {K.number_of_nodes()}, Social ties: {K.number_of_edges()}")
print(f"Two factions: Mr. Hi (instructor) vs Officer (administrator)")
print(f"True split: {sum(1 for v in true_labels.values() if v == 'Mr. Hi')} vs "
      f"{sum(1 for v in true_labels.values() if v == 'Officer')}")

# %% 2. Fiedler Vector Partition (Spectral Bisection)
L = nx.laplacian_matrix(K).toarray().astype(float)
eigenvalues, eigenvectors = eigh(L)
fiedler_vector = eigenvectors[:, 1]

spectral_labels = {i: ("Mr. Hi" if fiedler_vector[i] < 0 else "Officer")
                   for i in K.nodes()}

correct = sum(1 for n in K.nodes() if spectral_labels[n] == true_labels[n])
accuracy = correct / K.number_of_nodes()
# Try flipped labels too (spectral doesn't know which sign is which group)
accuracy = max(accuracy, 1 - accuracy)

print(f"\n--- Spectral Bisection (Fiedler vector) ---")
print(f"  Accuracy: {accuracy:.1%} ({correct if accuracy > 0.5 else K.number_of_nodes() - correct}"
      f"/{K.number_of_nodes()} correct)")

# %% 3. Modularity and Girvan-Newman
print("\n--- Girvan-Newman (Edge Betweenness) ---")
communities_gen = nx.community.girvan_newman(K)
top_level = next(communities_gen)  # first split into 2 communities
gn_groups = list(top_level)

gn_labels = {}
for i, group in enumerate(gn_groups):
    for node in group:
        gn_labels[node] = i

modularity = nx.community.modularity(K, gn_groups)
print(f"  Modularity Q = {modularity:.4f}")
print(f"  (Q > 0.3 generally indicates significant community structure)")

# %% 4. Louvain Algorithm
louvain_communities = nx.community.louvain_communities(K, seed=42)
louvain_modularity = nx.community.modularity(K, louvain_communities)
print(f"\n--- Louvain Algorithm ---")
print(f"  Communities found: {len(louvain_communities)}")
print(f"  Modularity Q = {louvain_modularity:.4f}")
for i, comm in enumerate(louvain_communities):
    print(f"  Community {i}: {sorted(comm)}")

# %% 5. Financial Analogy: Map Karate Club to a Market
print("\n" + "=" * 70)
print("FINANCIAL ANALOGY")
print("=" * 70)
print("""
  Zachary's Karate Club → Financial Market Structure

  MAPPING:
  ────────
  Club members      → Stocks / Assets / Institutions
  Social ties       → Return correlations / Exposure links
  Two factions      → Market regimes / Sector clusters
  Mr. Hi (node 0)   → "Growth" leader (e.g., tech mega-cap)
  Officer (node 33)  → "Value" leader (e.g., financial blue-chip)
  Club split         → Regime change / Sector rotation

  WHY THIS MATTERS:
  ─────────────────
  • Community detection on correlation networks → SECTOR DISCOVERY
    (without using SIC codes or GICS — purely data-driven)
  • Detecting the "split" early → REGIME CHANGE PREDICTION
  • Bridge nodes between communities → DIVERSIFICATION OPPORTUNITIES
  • Modularity → measures how "cleanly" the market partitions
    (high Q during crises = flight-to-quality clustering)
""")

# %% 6. Apply to a Real-ish Financial Network
# Build a stock correlation network that naturally has communities
np.random.seed(42)

sector_data = {
    "Tech":     ["AAPL", "MSFT", "GOOG", "NVDA", "META", "AMZN"],
    "Finance":  ["JPM", "GS", "BAC", "C", "MS", "WFC"],
    "Energy":   ["XOM", "CVX", "COP", "SLB", "EOG", "PSX"],
    "Health":   ["JNJ", "PFE", "UNH", "MRK", "ABT", "TMO"],
}

G_fin = nx.Graph()
for sector, tickers in sector_data.items():
    for t in tickers:
        G_fin.add_node(t, sector=sector)

# Intra-sector: strong correlations
for sector, tickers in sector_data.items():
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            corr = np.random.uniform(0.55, 0.90)
            G_fin.add_edge(tickers[i], tickers[j], weight=corr)

# Inter-sector: weak, sparse correlations
all_tickers = [t for ts in sector_data.values() for t in ts]
for _ in range(20):
    i, j = np.random.choice(len(all_tickers), 2, replace=False)
    s1 = G_fin.nodes[all_tickers[i]]["sector"]
    s2 = G_fin.nodes[all_tickers[j]]["sector"]
    if s1 != s2 and not G_fin.has_edge(all_tickers[i], all_tickers[j]):
        corr = np.random.uniform(0.10, 0.35)
        G_fin.add_edge(all_tickers[i], all_tickers[j], weight=corr)

print("\n--- Community Detection on Stock Correlation Network ---")
print(f"Stocks: {G_fin.number_of_nodes()}, Links: {G_fin.number_of_edges()}")

# Louvain on financial network
fin_communities = nx.community.louvain_communities(G_fin, weight="weight", seed=42)
fin_modularity = nx.community.modularity(G_fin, fin_communities, weight="weight")
print(f"Louvain modularity: {fin_modularity:.4f}")

print("\nDetected communities vs true sectors:")
for i, comm in enumerate(fin_communities):
    sectors_in = [G_fin.nodes[n]["sector"] for n in comm]
    dominant = max(set(sectors_in), key=sectors_in.count)
    purity = sectors_in.count(dominant) / len(sectors_in)
    print(f"  Community {i}: {sorted(comm)}")
    print(f"    → Dominant sector: {dominant} (purity: {purity:.0%})")

# %% 7. Bridge Nodes — Diversification Opportunities
print("\n--- Bridge Nodes (High Betweenness Between Communities) ---")
between_c = nx.betweenness_centrality(G_fin, weight="weight")
top_bridges = sorted(between_c.items(), key=lambda x: x[1], reverse=True)[:8]
print("  Stocks that bridge sector communities:")
for stock, bc in top_bridges:
    sector = G_fin.nodes[stock]["sector"]
    neighbors = list(G_fin.neighbors(stock))
    cross_sector = [n for n in neighbors if G_fin.nodes[n]["sector"] != sector]
    print(f"  {stock:6s} ({sector:8s}): betweenness={bc:.4f}, "
          f"cross-sector links: {cross_sector}")

# %% 8. Comprehensive Visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 16))

# --- Karate Club: True Labels ---
ax = axes[0, 0]
pos_k = nx.spring_layout(K, seed=42)
colors_true = ["#ef4444" if true_labels[n] == "Mr. Hi" else "#3b82f6" for n in K.nodes()]
nx.draw(K, pos_k, ax=ax, with_labels=True, node_color=colors_true,
        node_size=600, font_size=8, edge_color="gray", alpha=0.8)
ax.set_title("Karate Club — Ground Truth\n(Red=Mr. Hi, Blue=Officer)", fontsize=12)

# --- Karate Club: Spectral Partition ---
ax = axes[0, 1]
colors_fiedler = ["#ef4444" if fiedler_vector[n] < 0 else "#3b82f6" for n in K.nodes()]
sizes = 300 + 1500 * np.abs(fiedler_vector) / np.abs(fiedler_vector).max()
nx.draw(K, pos_k, ax=ax, with_labels=True, node_color=colors_fiedler,
        node_size=sizes, font_size=8, edge_color="gray", alpha=0.8)
ax.set_title(f"Karate Club — Fiedler Partition\n(Accuracy: {accuracy:.0%})", fontsize=12)

# --- Financial Network: True Sectors ---
ax = axes[1, 0]
pos_fin = nx.spring_layout(G_fin, seed=42, k=1.5)
sector_colors = {"Tech": "#3b82f6", "Finance": "#ef4444",
                 "Energy": "#f59e0b", "Health": "#10b981"}
fin_colors = [sector_colors[G_fin.nodes[n]["sector"]] for n in G_fin.nodes()]
nx.draw(G_fin, pos_fin, ax=ax, with_labels=True, node_color=fin_colors,
        node_size=800, font_size=8, font_weight="bold",
        edge_color="gray", alpha=0.7, width=0.5)
for sector, color in sector_colors.items():
    ax.scatter([], [], c=color, s=100, label=sector)
ax.legend(loc="lower left", fontsize=9)
ax.set_title(f"Stock Network — True Sectors", fontsize=12)

# --- Financial Network: Detected Communities ---
ax = axes[1, 1]
community_colors_map = {}
cmap = plt.cm.Set1
for i, comm in enumerate(fin_communities):
    for node in comm:
        community_colors_map[node] = cmap(i / max(len(fin_communities) - 1, 1))
det_colors = [community_colors_map[n] for n in G_fin.nodes()]
nx.draw(G_fin, pos_fin, ax=ax, with_labels=True, node_color=det_colors,
        node_size=800, font_size=8, font_weight="bold",
        edge_color="gray", alpha=0.7, width=0.5)
ax.set_title(f"Stock Network — Louvain Communities\n(Q = {fin_modularity:.3f})", fontsize=12)

plt.tight_layout()
plt.savefig("/mnt/nas/Class/2026/PHD/networkX/05_karate_club_finance.png", dpi=150)
plt.show()
print("\nFigure saved: 05_karate_club_finance.png")

# %% 9. Modularity Over Time — Regime Detection
print("\n--- Modularity as Regime Indicator ---")
print("  Simulating market stress: increasing cross-sector correlations\n")

stress_levels = np.linspace(0, 0.8, 9)
modularities = []

for stress in stress_levels:
    G_stress = G_fin.copy()
    for u, v, d in G_stress.edges(data=True):
        s1 = G_stress.nodes[u]["sector"]
        s2 = G_stress.nodes[v]["sector"]
        if s1 != s2:
            # During stress, cross-sector correlations rise
            d["weight"] = min(d["weight"] + stress, 0.95)
    comms = nx.community.louvain_communities(G_stress, weight="weight", seed=42)
    Q = nx.community.modularity(G_stress, comms, weight="weight")
    modularities.append(Q)
    print(f"  Stress = {stress:.1f}: Q = {Q:.4f}, Communities = {len(comms)}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(stress_levels, modularities, "o-", color="#ef4444", linewidth=2, markersize=8)
ax.set_xlabel("Market Stress Level (cross-sector correlation increase)", fontsize=12)
ax.set_ylabel("Modularity Q", fontsize=12)
ax.set_title("Modularity Collapse Under Market Stress\n"
             "(Sectors blur → everything correlated → no community structure)",
             fontsize=13)
ax.axhline(0.3, color="gray", linestyle="--", alpha=0.5, label="Significant structure (Q=0.3)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/mnt/nas/Class/2026/PHD/networkX/05_modularity_stress.png", dpi=150)
plt.show()
print("\nFigure saved: 05_modularity_stress.png")

# %% 10. Exercises
print("\n" + "=" * 70)
print("EXERCISES")
print("=" * 70)
print("""
1. On the Karate Club: compare Fiedler partition, Girvan-Newman,
   and Louvain. Which achieves the best accuracy? Why might
   Girvan-Newman outperform spectral methods here?

2. Compute the NORMALIZED CUT of the Fiedler partition:
   Ncut = cut(A,B)/vol(A) + cut(A,B)/vol(B)
   where cut = total weight of cross-group edges,
   vol = total degree within group.
   Lower Ncut → better partition.

3. Build a stock correlation network from real daily returns
   (e.g., S&P 500 from Yahoo Finance). Apply Louvain.
   Do detected communities match GICS sectors?
   How does modularity change between 2019 (calm) and 2020 (COVID)?

4. Implement a simple CONTAGION MODEL on the Karate Club:
   - Start: node 0 is "defaulted"
   - Each step: neighbors of defaulted nodes default with probability
     proportional to edge weight
   - Question: does the contagion cross the community boundary?
     How many steps to reach node 33?

5. (Research) Read Cont & Schaanning (2019) "Monitoring indirect
   contagion" — how does community structure in the banking network
   relate to fire-sale externalities?
""")
