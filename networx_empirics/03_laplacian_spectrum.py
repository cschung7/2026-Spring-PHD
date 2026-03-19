"""
03 - Graph Laplacian and Spectral Analysis for Finance
======================================================
PhD Finance | Network Analysis

The Laplacian matrix is the cornerstone of spectral graph theory.
In finance: portfolio clustering, systemic risk measurement,
and Fiedler eigenvalue as a "network fragility" indicator.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# %% 1. Build a Sovereign Debt Exposure Network
# Nodes = countries, edges = bilateral debt exposures
# Inspired by BIS cross-border banking statistics

G = nx.Graph()

countries = {
    "US": "Americas", "UK": "Europe", "DE": "Europe", "FR": "Europe",
    "JP": "Asia", "CN": "Asia", "KR": "Asia",
    "BR": "Americas", "IT": "Europe", "ES": "Europe",
}
for c, region in countries.items():
    G.add_node(c, region=region)

# Bilateral exposures ($B) — symmetric for simplicity
exposures = [
    ("US", "UK", 450), ("US", "JP", 380), ("US", "DE", 290),
    ("US", "CN", 180), ("US", "KR", 120), ("US", "BR", 95),
    ("UK", "DE", 320), ("UK", "FR", 280), ("UK", "JP", 150),
    ("UK", "IT", 110), ("UK", "ES", 90),
    ("DE", "FR", 350), ("DE", "IT", 200), ("DE", "ES", 160),
    ("DE", "JP", 85),
    ("FR", "IT", 240), ("FR", "ES", 180), ("FR", "JP", 70),
    ("JP", "CN", 130), ("JP", "KR", 170),
    ("CN", "KR", 95),  ("CN", "BR", 60),
    ("KR", "US", 140),  # already US-KR exists, this makes it undirected
    ("IT", "ES", 130),
    ("BR", "US", 95),   # duplicate handled by graph
]
for u, v, w in exposures:
    if G.has_edge(u, v):
        G[u][v]["weight"] += w
    else:
        G.add_edge(u, v, weight=w)

print("=" * 70)
print("SOVEREIGN DEBT EXPOSURE NETWORK — LAPLACIAN ANALYSIS")
print("=" * 70)
print(f"Countries: {G.number_of_nodes()}, Bilateral links: {G.number_of_edges()}")

# %% 2. Adjacency and Laplacian Matrices
nodes = list(G.nodes())
n = len(nodes)

A = nx.adjacency_matrix(G, nodelist=nodes, weight="weight").toarray().astype(float)
D = np.diag(A.sum(axis=1))  # Degree matrix (weighted)
L = D - A                    # Graph Laplacian: L = D - A

print("\n--- Degree Matrix (diagonal = total exposure per country, $B) ---")
for i, c in enumerate(nodes):
    print(f"  {c}: ${D[i, i]:.0f}B total bilateral exposure")

print("\n--- Laplacian Properties ---")
print(f"  L is symmetric:      {np.allclose(L, L.T)}")
print(f"  Row sums are zero:   {np.allclose(L.sum(axis=1), 0)}")
print(f"  L is positive semi-definite (all eigenvalues ≥ 0)")

# %% 3. Eigendecomposition of the Laplacian
eigenvalues, eigenvectors = eigh(L)  # eigh for symmetric matrices

print("\n--- Laplacian Spectrum ---")
for i, lam in enumerate(eigenvalues):
    print(f"  λ_{i} = {lam:12.2f}", end="")
    if i == 0:
        print("  ← always 0 (connected graph)")
    elif i == 1:
        print("  ← FIEDLER VALUE (algebraic connectivity)")
    elif i == n - 1:
        print("  ← largest eigenvalue (spectral radius)")
    else:
        print()

# %% 4. Fiedler Value — Algebraic Connectivity
fiedler_value = eigenvalues[1]
fiedler_vector = eigenvectors[:, 1]

print("\n" + "=" * 70)
print("FIEDLER VALUE (λ₂) — ALGEBRAIC CONNECTIVITY")
print("=" * 70)
print(f"""
  λ₂ = {fiedler_value:.2f}

  FINANCIAL INTERPRETATION:
  ─────────────────────────
  • λ₂ measures how DIFFICULT it is to split the network into two groups
  • HIGHER λ₂ → more robust financial system, harder to fragment
  • LOWER λ₂  → fragile system, a few link removals disconnect it
  • λ₂ = 0    → network is ALREADY disconnected (two components)

  • In sovereign debt: low λ₂ means the global financial system
    has a "fault line" — a natural partition into two blocs
    that could decouple during a crisis

  • Monitoring λ₂ over time → early warning for financial fragmentation
""")

# %% 5. Fiedler Vector — Natural Partition of the Financial System
print("--- Fiedler Vector (partition indicator) ---")
print("  Countries sorted by Fiedler vector component:")
sorted_idx = np.argsort(fiedler_vector)
for i in sorted_idx:
    sign = "+" if fiedler_vector[i] >= 0 else "−"
    bar = "█" * int(abs(fiedler_vector[i]) * 80)
    region = countries[nodes[i]]
    print(f"  {nodes[i]:4s} ({region:8s}): {sign}{abs(fiedler_vector[i]):.4f}  {bar}")

print("\n  → Countries with SAME SIGN cluster together")
print("  → The sign boundary reveals the network's natural fault line")
group_pos = [nodes[i] for i in range(n) if fiedler_vector[i] >= 0]
group_neg = [nodes[i] for i in range(n) if fiedler_vector[i] < 0]
print(f"\n  Group A (positive): {group_pos}")
print(f"  Group B (negative): {group_neg}")

# %% 6. Normalized Laplacian — accounts for different country sizes
L_norm = nx.normalized_laplacian_matrix(G, nodelist=nodes, weight="weight").toarray()
evals_norm, evecs_norm = eigh(L_norm)

print("\n--- Normalized Laplacian Spectrum ---")
for i, lam in enumerate(evals_norm):
    print(f"  λ̃_{i} = {lam:.6f}", end="")
    if i == 0:
        print("  ← always 0")
    elif i == 1:
        print(f"  ← normalized Fiedler value")
    else:
        print()

print(f"\n  Normalized λ₂ = {evals_norm[1]:.6f}")
print("  (Normalized version accounts for heterogeneous node degrees)")
print("  (More appropriate when country exposures vary widely)")

# %% 7. Spectral Gap and Mixing Time
spectral_gap = evals_norm[1]
print(f"\n--- Spectral Gap = {spectral_gap:.6f} ---")
print(f"  Mixing time ∝ 1/spectral_gap ≈ {1/spectral_gap:.1f}")
print("  → How quickly a shock in one country spreads to the ENTIRE system")
print("  → Smaller gap = slower mixing = more isolated clusters")
print("  → Larger gap = faster mixing = rapid contagion")

# %% 8. Visualization
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Network with Fiedler coloring
ax = axes[0]
pos = nx.spring_layout(G, seed=42, k=2)
node_colors = ["#ef4444" if fiedler_vector[i] < 0 else "#3b82f6"
               for i in range(n)]
weights = [G[u][v]["weight"] for u, v in G.edges()]
max_w = max(weights)
widths = [0.5 + 4 * w / max_w for w in weights]

nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
        node_size=1500, font_size=10, font_weight="bold",
        width=widths, edge_color="gray", alpha=0.8)
ax.set_title("Fiedler Partition of Sovereign Debt Network\n"
             "(Red vs Blue = natural fault line)", fontsize=11)

# Eigenvalue spectrum
ax = axes[1]
ax.bar(range(n), eigenvalues, color=["green" if i == 1 else "steelblue"
       for i in range(n)], edgecolor="black")
ax.set_xlabel("Index", fontsize=11)
ax.set_ylabel("Eigenvalue (λ)", fontsize=11)
ax.set_title(f"Laplacian Spectrum\n(Fiedler value λ₂ = {fiedler_value:.1f})", fontsize=11)
ax.set_xticks(range(n))
ax.set_xticklabels([f"λ_{i}" for i in range(n)], fontsize=9)

# Fiedler vector bar chart
ax = axes[2]
order = np.argsort(fiedler_vector)
colors = ["#ef4444" if fiedler_vector[i] < 0 else "#3b82f6" for i in order]
ax.barh(range(n), fiedler_vector[order], color=colors, edgecolor="black")
ax.set_yticks(range(n))
ax.set_yticklabels([nodes[i] for i in order], fontsize=10)
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("Fiedler Vector Component", fontsize=11)
ax.set_title("Fiedler Vector — Country Partition", fontsize=11)

plt.tight_layout()
plt.savefig("/mnt/nas/Class/2026/PHD/networkX/03_laplacian_spectrum.png", dpi=150)
plt.show()
print("\nFigure saved: 03_laplacian_spectrum.png")

# %% 9. Stress Test: Remove an Edge and Watch λ₂
print("\n--- Stress Test: Effect of Link Removal on Algebraic Connectivity ---")
print(f"  Baseline λ₂ = {fiedler_value:.2f}\n")

edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)
for u, v, d in edges_sorted[:5]:
    G_temp = G.copy()
    G_temp.remove_edge(u, v)
    L_temp = nx.laplacian_matrix(G_temp, nodelist=nodes, weight="weight").toarray().astype(float)
    evals_temp = np.sort(eigh(L_temp, eigvals_only=True))
    delta = evals_temp[1] - fiedler_value
    print(f"  Remove {u}-{v} (${d['weight']:.0f}B): "
          f"λ₂ = {evals_temp[1]:.2f}  (Δ = {delta:+.2f})")

print("\n  → Edges whose removal drops λ₂ the most are CRITICAL LINKS")
print("  → These are the connections whose failure would most fragment the system")

# %% 10. Exercises
print("\n" + "=" * 70)
print("EXERCISES")
print("=" * 70)
print("""
1. Compute the number of spanning trees using Kirchhoff's theorem:
   τ(G) = (1/n) × Π(λ_i) for i = 1..n-1
   What does a high number of spanning trees mean financially?

2. Track λ₂ as you progressively remove the weakest edges (lowest weight).
   Plot λ₂ vs number of edges removed. Where does λ₂ drop to zero?
   This is the "percolation threshold" — the fragmentation point.

3. Use the Fiedler vector to perform spectral clustering into k=3 groups.
   (Hint: use the 2nd and 3rd eigenvectors as coordinates, then k-means)
   Do the clusters correspond to geographic regions?

4. Compare L = D - A (combinatorial) vs L_sym = D^{-1/2} L D^{-1/2}
   (symmetric normalized). When does normalization matter?
   (Hint: think about US vs KR in terms of total exposure)

5. (Research) Read Diebold & Yilmaz (2014) "On the network topology of
   variance decompositions" — how does their connectedness index
   relate to the spectral properties of the Laplacian?
""")
