"""
01 - Graphs, Nodes, and Edges: Financial Network Fundamentals
=============================================================
PhD Finance | Network Analysis

Concept: Banks, firms, and institutions as nodes; loans, trades,
         and ownership links as edges.
"""

import networkx as nx
import matplotlib.pyplot as plt

# %% 1. Interbank Lending Network (Directed, Weighted)
# Each edge = a loan from bank A to bank B with amount (weight)

G_bank = nx.DiGraph()

# Banks
banks = ["JPMorgan", "Goldman", "Citi", "BofA", "Morgan Stanley", "Wells Fargo"]
G_bank.add_nodes_from(banks)

# Interbank loans: (lender, borrower, amount in $B)
loans = [
    ("JPMorgan", "Goldman", 12.5),
    ("JPMorgan", "Citi", 8.3),
    ("Goldman", "Morgan Stanley", 6.7),
    ("Goldman", "BofA", 4.2),
    ("Citi", "BofA", 9.1),
    ("Citi", "Wells Fargo", 3.8),
    ("BofA", "JPMorgan", 7.6),
    ("Morgan Stanley", "Citi", 5.4),
    ("Wells Fargo", "JPMorgan", 2.9),
    ("Wells Fargo", "Goldman", 4.1),
]

for lender, borrower, amount in loans:
    G_bank.add_edge(lender, borrower, weight=amount)

print("=" * 60)
print("INTERBANK LENDING NETWORK")
print("=" * 60)
print(f"Banks (nodes):        {G_bank.number_of_nodes()}")
print(f"Loan links (edges):   {G_bank.number_of_edges()}")
print(f"Directed:             {G_bank.is_directed()}")

# %% 2. Basic Graph Properties
print("\n--- Degree Analysis ---")
for bank in banks:
    in_deg = G_bank.in_degree(bank)    # borrowing from others
    out_deg = G_bank.out_degree(bank)  # lending to others
    in_vol = sum(d["weight"] for _, _, d in G_bank.in_edges(bank, data=True))
    out_vol = sum(d["weight"] for _, _, d in G_bank.out_edges(bank, data=True))
    print(f"{bank:16s} | Lends to: {out_deg} banks (${out_vol:.1f}B) | "
          f"Borrows from: {in_deg} banks (${in_vol:.1f}B)")

# %% 3. Density — how interconnected is the system?
density = nx.density(G_bank)
print(f"\nNetwork density: {density:.3f}")
print(f"  (1.0 = every bank lends to every other bank)")
print(f"  (Higher density → more systemic risk contagion paths)")

# %% 4. Adjacency Matrix — foundation for financial network analysis
import numpy as np

A = nx.adjacency_matrix(G_bank, weight="weight").toarray()
print("\n--- Weighted Adjacency Matrix ($B) ---")
print(f"{'':16s}", end="")
for b in banks:
    print(f"{b[:6]:>8s}", end="")
print()
for i, b in enumerate(banks):
    print(f"{b:16s}", end="")
    for j in range(len(banks)):
        val = A[i, j]
        print(f"{val:8.1f}" if val > 0 else f"{'—':>8s}", end="")
    print()

# %% 5. Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Unweighted view
pos = nx.spring_layout(G_bank, seed=42)
ax = axes[0]
nx.draw(G_bank, pos, ax=ax, with_labels=True, node_color="lightblue",
        node_size=2000, font_size=9, font_weight="bold",
        arrows=True, arrowsize=20, edge_color="gray")
ax.set_title("Interbank Lending Network (Topology)", fontsize=13)

# Weighted view — edge width proportional to loan size
ax = axes[1]
weights = [G_bank[u][v]["weight"] for u, v in G_bank.edges()]
max_w = max(weights)
widths = [3 * w / max_w for w in weights]

nx.draw(G_bank, pos, ax=ax, with_labels=True, node_color="salmon",
        node_size=2000, font_size=9, font_weight="bold",
        arrows=True, arrowsize=20, edge_color="gray",
        width=widths)
edge_labels = {(u, v): f"${d['weight']}B" for u, v, d in G_bank.edges(data=True)}
nx.draw_networkx_edge_labels(G_bank, pos, edge_labels, font_size=7, ax=ax)
ax.set_title("Interbank Lending Network (Weighted by Loan Size)", fontsize=13)

plt.tight_layout()
plt.savefig("/mnt/nas/Class/2026/PHD/networkX/01_interbank_network.png", dpi=150)
plt.show()
print("\nFigure saved: 01_interbank_network.png")

# %% 6. Exercise
print("\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
1. Add 'Deutsche Bank' to the network with 3 lending relationships.
   How does density change?

2. Which bank is the biggest net lender? Net borrower?
   (Hint: out_volume - in_volume)

3. Remove JPMorgan from the network. How many connected components
   remain? What does this imply about systemic importance?

4. Create an undirected version: G_undirected = G_bank.to_undirected()
   Compare density of directed vs undirected. Why do they differ?
""")
