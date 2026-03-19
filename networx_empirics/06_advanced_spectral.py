"""
06 - Advanced Spectral Methods: From Random Walks to Systemic Risk
=================================================================
PhD Finance | Network Analysis

Connecting the Laplacian, random walks, diffusion, and
Markov chains — all foundational for financial contagion modeling.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, expm

# %% 1. Build a Credit Default Swap (CDS) Network
# Nodes = major financial institutions
# Edges = bilateral CDS exposures (protection buyer ↔ seller)

institutions = {
    "JPM": "US-Bank", "GS": "US-Bank", "MS": "US-Bank",
    "Citi": "US-Bank", "BAC": "US-Bank",
    "HSBC": "EU-Bank", "DB": "EU-Bank", "BNP": "EU-Bank",
    "AIG": "Insurance", "MetLife": "Insurance",
    "Lehman": "US-Bank",  # the one that failed
}

G = nx.Graph()
for inst, itype in institutions.items():
    G.add_node(inst, type=itype)

# CDS exposures ($B notional)
cds_links = [
    ("JPM", "GS", 85), ("JPM", "MS", 60), ("JPM", "Citi", 95),
    ("JPM", "HSBC", 40), ("JPM", "AIG", 110),
    ("GS", "MS", 70), ("GS", "DB", 55), ("GS", "AIG", 90),
    ("GS", "Lehman", 75),
    ("MS", "Citi", 45), ("MS", "Lehman", 50), ("MS", "MetLife", 30),
    ("Citi", "BAC", 80), ("Citi", "BNP", 35), ("Citi", "AIG", 65),
    ("BAC", "AIG", 55), ("BAC", "MetLife", 40),
    ("HSBC", "DB", 60), ("HSBC", "BNP", 50),
    ("DB", "BNP", 45), ("DB", "Lehman", 40),
    ("BNP", "AIG", 25),
    ("AIG", "MetLife", 35), ("AIG", "Lehman", 100),  # AIG-Lehman: massive
    ("Lehman", "Citi", 55),
]
for u, v, w in cds_links:
    G.add_edge(u, v, weight=w)

nodes = list(G.nodes())
n = len(nodes)

print("=" * 70)
print("CDS EXPOSURE NETWORK — ADVANCED SPECTRAL ANALYSIS")
print("=" * 70)
print(f"Institutions: {n}, CDS links: {G.number_of_edges()}")

# %% 2. Transition Matrix (Random Walk on the Network)
A = nx.adjacency_matrix(G, nodelist=nodes, weight="weight").toarray().astype(float)
D = np.diag(A.sum(axis=1))
D_inv = np.diag(1.0 / A.sum(axis=1))

# Random walk transition matrix: P = D^{-1} A
# P[i,j] = probability of walking from i to j (proportional to edge weight)
P = D_inv @ A

print("\n--- Random Walk Transition Matrix ---")
print("  P[i,j] = Prob(walk from i to j) ∝ CDS exposure weight")
print("  Financial meaning: probability that a shock at i propagates to j\n")

print(f"{'':>8s}", end="")
for inst in nodes:
    print(f"{inst:>8s}", end="")
print()
for i, inst in enumerate(nodes):
    print(f"{inst:>8s}", end="")
    for j in range(n):
        print(f"{P[i, j]:8.3f}", end="")
    print()

# %% 3. Stationary Distribution (long-run shock concentration)
# For undirected graphs: π_i = d_i / 2m (degree-proportional)
total_weight = sum(d["weight"] for _, _, d in G.edges(data=True))
stationary = {nodes[i]: D[i, i] / (2 * total_weight) for i in range(n)}

print("\n--- Stationary Distribution (Long-Run Shock Concentration) ---")
print("  π_i = where shocks accumulate after many propagation steps\n")
for inst, prob in sorted(stationary.items(), key=lambda x: x[1], reverse=True):
    bar = "█" * int(prob * 200)
    print(f"  {inst:>8s}: {prob:.4f} {bar}")

print("\n  → Institutions with highest π absorb the most risk in the long run")
print("  → These are the 'risk sinks' of the financial system")

# %% 4. Heat Diffusion on the Network
# Solution to ∂u/∂t = -L·u is u(t) = exp(-tL)·u(0)
# Models how a shock at one node diffuses through the network

L = nx.laplacian_matrix(G, nodelist=nodes, weight="weight").toarray().astype(float)

# Shock at Lehman (node index)
lehman_idx = nodes.index("Lehman")

u0 = np.zeros(n)
u0[lehman_idx] = 1.0  # unit shock at Lehman

print("\n--- Heat Diffusion: Lehman Shock Propagation ---")
print("  u(t) = exp(-tL) · u(0)")
print("  Time = number of propagation steps\n")

times = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
diffusion_results = {}

print(f"{'Time':>8s}", end="")
for inst in nodes:
    print(f"{inst:>8s}", end="")
print()

for t in times:
    u_t = expm(-t * L) @ u0
    diffusion_results[t] = u_t
    print(f"{t:8.3f}", end="")
    for j in range(n):
        print(f"{u_t[j]:8.4f}", end="")
    print()

print("\n  → At t=0: all shock is at Lehman")
print("  → As t→∞: shock spreads to stationary distribution")
print("  → Rate of spread governed by λ₂ (Fiedler value)")

# %% 5. Communicability — Estrada Index
# Communicability between i and j = (exp(A))_{ij}
# Counts ALL walks, weighted by 1/k! for walks of length k
# Financial meaning: total indirect exposure through all channels

expA = expm(A / np.max(A))  # normalize to prevent overflow
print("\n--- Communicability Matrix (top pairs) ---")
print("  (exp(A))_{ij} = total weighted walk count between i and j")
print("  Captures INDIRECT exposure through all intermediary chains\n")

pairs = []
for i in range(n):
    for j in range(i + 1, n):
        pairs.append((nodes[i], nodes[j], expA[i, j]))

pairs.sort(key=lambda x: x[2], reverse=True)
print("  Most 'communicable' pairs (highest indirect exposure):")
for u, v, comm in pairs[:10]:
    direct = A[nodes.index(u), nodes.index(v)]
    print(f"  {u:>8s} ↔ {v:<8s}: communicability = {comm:.4f} "
          f"(direct = ${direct:.0f}B)")

# %% 6. Effective Resistance — Network Robustness
# R_eff(i,j) = (e_i - e_j)^T L^+ (e_i - e_j)
# where L^+ is the Moore-Penrose pseudoinverse of L
# Financial meaning: how "electrically" resistant is the path from i to j?
# Low resistance = many redundant paths = harder to disrupt

eigenvalues, eigenvectors = eigh(L)
# Pseudoinverse: L^+ = V Λ^+ V^T (skip zero eigenvalue)
L_pinv = eigenvectors[:, 1:] @ np.diag(1.0 / eigenvalues[1:]) @ eigenvectors[:, 1:].T

print("\n--- Effective Resistance (Network Robustness Metric) ---")
print("  Low R_eff = many redundant paths = robust connection")
print("  High R_eff = few paths = fragile connection\n")

for u, v, _ in pairs[:8]:
    i, j = nodes.index(u), nodes.index(v)
    e_diff = np.zeros(n)
    e_diff[i] = 1
    e_diff[j] = -1
    R_eff = e_diff @ L_pinv @ e_diff
    print(f"  {u:>8s} ↔ {v:<8s}: R_eff = {R_eff:.6f}")

# Total effective resistance (Kirchhoff index)
kirchhoff = n * np.sum(1.0 / eigenvalues[1:])
print(f"\n  Kirchhoff index (total resistance) = {kirchhoff:.2f}")
print("  → Lower = more robust network overall")
print("  → Can be monitored over time as a systemic risk indicator")

# %% 7. Visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 16))

pos = nx.spring_layout(G, seed=42, k=2)
type_colors = {"US-Bank": "#3b82f6", "EU-Bank": "#f59e0b",
               "Insurance": "#10b981"}

# Network topology
ax = axes[0, 0]
node_colors = [type_colors[G.nodes[n]["type"]] for n in nodes]
weights = [G[u][v]["weight"] for u, v in G.edges()]
widths = [0.5 + 3 * w / max(weights) for w in weights]
nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
        node_size=1200, font_size=9, font_weight="bold",
        width=widths, edge_color="gray", alpha=0.8)
for t, c in type_colors.items():
    ax.scatter([], [], c=c, s=100, label=t)
ax.legend(loc="lower left", fontsize=9)
ax.set_title("CDS Exposure Network", fontsize=13)

# Stationary distribution
ax = axes[0, 1]
stat_vals = [stationary[inst] for inst in nodes]
sizes = [500 + 4000 * s / max(stat_vals) for s in stat_vals]
nx.draw(G, pos, ax=ax, with_labels=True, node_color=stat_vals,
        cmap="YlOrRd", node_size=sizes, font_size=9, font_weight="bold",
        edge_color="gray", alpha=0.7, width=0.5)
ax.set_title("Stationary Distribution\n(Larger = absorbs more long-run risk)", fontsize=13)

# Diffusion at different times
ax = axes[1, 0]
t_plot = 0.01
u_plot = diffusion_results[t_plot]
sizes = [300 + 3000 * abs(u) for u in u_plot]
nx.draw(G, pos, ax=ax, with_labels=True, node_color=u_plot,
        cmap="Reds", node_size=sizes, font_size=9, font_weight="bold",
        edge_color="gray", alpha=0.7, width=0.5, vmin=0, vmax=1)
ax.set_title(f"Lehman Shock Diffusion (t={t_plot})\n(Darker = more shock absorbed)", fontsize=13)

# Laplacian spectrum
ax = axes[1, 1]
colors = ["green" if i == 1 else "steelblue" for i in range(n)]
colors[0] = "lightgray"
ax.bar(range(n), eigenvalues, color=colors, edgecolor="black")
ax.set_xlabel("Index", fontsize=11)
ax.set_ylabel("Eigenvalue", fontsize=11)
ax.set_title(f"Laplacian Spectrum\n(λ₂ = {eigenvalues[1]:.1f} = algebraic connectivity)", fontsize=13)
ax.set_xticks(range(n))
ax.set_xticklabels([f"λ_{i}" for i in range(n)], fontsize=8)

plt.tight_layout()
plt.savefig("/mnt/nas/Class/2026/PHD/networkX/06_advanced_spectral.png", dpi=150)
plt.show()
print("\nFigure saved: 06_advanced_spectral.png")

# %% 8. Shock Simulation: Remove Lehman
print("\n--- Systemic Impact: Remove Lehman ---")
G_post = G.copy()
G_post.remove_node("Lehman")
nodes_post = list(G_post.nodes())

print(f"  Before: {G.number_of_nodes()} institutions, {G.number_of_edges()} links")
print(f"  After:  {G_post.number_of_nodes()} institutions, {G_post.number_of_edges()} links")
print(f"  Connected: {nx.is_connected(G_post)}")

L_post = nx.laplacian_matrix(G_post, nodelist=nodes_post, weight="weight").toarray().astype(float)
evals_post = np.sort(eigh(L_post, eigvals_only=True))

print(f"\n  λ₂ before Lehman removal: {eigenvalues[1]:.2f}")
print(f"  λ₂ after  Lehman removal: {evals_post[1]:.2f}")
delta = evals_post[1] - eigenvalues[1]
print(f"  Change: {delta:+.2f} ({delta/eigenvalues[1]:+.1%})")
print("  → Negative change = network became MORE fragile")
print("  → Positive change = network became MORE robust (bottleneck removed)")

# Change in eigenvector centrality
ec_before = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
ec_after = nx.eigenvector_centrality(G_post, weight="weight", max_iter=1000)

print("\n  Eigenvector centrality changes after Lehman removal:")
for inst in nodes_post:
    delta_ec = ec_after[inst] - ec_before[inst]
    if abs(delta_ec) > 0.01:
        print(f"    {inst:>8s}: {ec_before[inst]:.4f} → {ec_after[inst]:.4f} ({delta_ec:+.4f})")

# %% 9. Exercises
print("\n" + "=" * 70)
print("EXERCISES")
print("=" * 70)
print("""
1. Compute the MEAN FIRST PASSAGE TIME from each institution to every
   other. Which pair has the longest MFPT? What does this mean for
   contagion speed?
   (Hint: MFPT m_{ij} = 1/π_j for random walks on undirected graphs
    is only the hitting time from stationarity. For exact MFPT, use
    the fundamental matrix Z = (I - P + Π)^{-1})

2. Compute the RETURN TIME for each institution:
   E[T_i→i] = 1/π_i
   Which institution has the shortest return time? Why?

3. Simulate the heat diffusion for different initial conditions:
   a) Shock at AIG (the insurance giant)
   b) Simultaneous shock at all EU banks
   c) Uniform small shock everywhere
   Compare the equilibrium distributions.

4. Compute the CHEEGER CONSTANT h(G) and compare with the Cheeger
   inequality: λ₂/2 ≤ h(G) ≤ √(2·λ₂)
   What does the Cheeger constant tell us about the "bottleneck"
   of the financial network?

5. (Research) Read Battiston et al. (2016) "The price of complexity
   in financial networks" — how do they use spectral methods to
   measure systemic risk? Compare their approach with the metrics
   computed in this exercise.
""")
