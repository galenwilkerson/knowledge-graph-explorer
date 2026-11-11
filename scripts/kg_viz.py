"""Export a LightRAG graph for the interactive viewer (viz/index.html). CPU only.

Usage (kg env):  python kg_viz.py <lightrag_dir> <out_json> [--iters 400]

Entities with no links are left out of the drawing (counted in the stats). Layout: DrL (igraph) as a start, then
ForceAtlas2 in PyTorch on the CPU: degree-weighted repulsion between all pairs (exact, computed in blocks),
linear attraction along edges, gravity toward the centre, and a cooling step size. Communities: Louvain.
Also computes the graph and KG statistics the viewer's panel shows, and the eval set's answer coverage.
"""
import json
import os
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

import igraph as ig
import networkx as nx
import numpy as np
import torch

SEP = "<SEP>"
DATA = Path(os.environ.get("KG_DATA", Path(__file__).resolve().parent.parent / "data"))
torch.set_num_threads(14)


def first(s, n):
    s = unquote((s or "").split(SEP)[0]).replace("_", " ").strip()
    return s if len(s) <= n else s[:n - 1] + "…"


def srcs(a):
    return [c for c in (a.get("file_path") or "").split(SEP) if c]


def norm(s):
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s.strip())
    s = re.sub(r"[-–—/_]", " ", s.lower())
    s = " ".join(re.sub(r"[^\w\s]", "", s).split())
    return s[:-1] if s.endswith("s") and len(s) > 3 else s


def forceatlas2(pos, edges, deg, iters, kr=1.0, kg=1.0, block=2048):
    """ForceAtlas2-style layout (Jacomy et al. 2014), exact O(n^2) repulsion in blocks."""
    x = torch.tensor(pos, dtype=torch.float32)
    x = (x - x.mean(0)) / x.std() * math.sqrt(len(x))            # start at a sensible scale
    mass = torch.tensor(deg, dtype=torch.float32) + 1
    e = torch.tensor(edges, dtype=torch.long)
    n = len(x)
    for it in range(iters):
        f = torch.zeros_like(x)
        for i in range(0, n, block):
            d = x[i:i + block, None, :] - x[None, :, :]              # (b, n, 2)
            d2 = (d * d).sum(-1).clamp_min(1e-2)
            f[i:i + block] += (d * (kr * mass[i:i + block, None] * mass[None, :] / d2)[..., None]).sum(1)
        delta = x[e[:, 0]] - x[e[:, 1]]                              # linear attraction along edges
        f.index_add_(0, e[:, 0], -delta)
        f.index_add_(0, e[:, 1], delta)
        r = x.norm(dim=1, keepdim=True).clamp_min(1e-3)
        f -= kg * mass[:, None] * x / r                              # gravity keeps small components near
        temp = 0.15 * math.sqrt(n) * (1 - it / iters) + 0.5           # cooling: max move per node per step
        step = f.norm(dim=1, keepdim=True).clamp_min(1e-9)
        x += f / step * torch.minimum(step / mass[:, None], torch.tensor(temp))
        if it % 50 == 0:
            print(f"  iter {it}: median move {torch.minimum(step / mass[:, None], torch.tensor(temp)).median():.3f}", flush=True)
    return x.numpy()


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    iters = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 400
    G = nx.read_graphml(src / "graph_chunk_entity_relation.graphml")
    n_all, m_all = G.number_of_nodes(), G.number_of_edges()
    deg_all = np.array([d for _, d in G.degree()])
    names_all = list(G.nodes())
    isolated = [n for n, d in G.degree() if d == 0]
    g = G.copy()
    g.remove_nodes_from(isolated)
    names = list(g.nodes())
    idx = {n: i for i, n in enumerate(names)}
    elist = [(idx[u], idx[v]) for u, v in g.edges() if u != v]
    deg = np.array([g.degree(n) for n in names])

    t0 = time.time()
    h = ig.Graph(n=len(names), edges=elist)
    cache = out.with_suffix(".layout.json")          # reuse a saved layout if the node set is unchanged
    saved = json.load(open(cache)) if cache.exists() and "--relayout" not in sys.argv else None
    if saved and saved["names"] == names:
        pos = np.array(saved["pos"])
        print(f"layout reused from {cache}", flush=True)
    else:
        pos = forceatlas2(np.array(h.layout_drl().coords), elist, deg, iters)
        json.dump({"names": names, "pos": pos.round(2).tolist(), "iters": iters}, open(cache, "w"))
        print(f"layout {time.time() - t0:.0f}s for {len(names)} nodes, saved to {cache}", flush=True)
    comm = h.community_multilevel()
    memb = comm.membership

    # --- statistics for the panel
    comps = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
    gcc_set = max(nx.connected_components(g), key=len)
    gc = g.subgraph(gcc_set)
    gnames = list(gcc_set)
    rng = random.Random(0)
    dists = [d for s in rng.sample(gnames, 150) for d in nx.single_source_shortest_path_length(gc, s).values() if d > 0]
    k = deg[deg > 0]
    tail = k[k >= 3]
    lab = Counter(w.strip().lower() for _, _, d in g.edges(data=True) for w in (d.get("keywords") or "").split(",") if w.strip())
    per_edge = np.array([len(srcs(d)) for _, _, d in g.edges(data=True)])
    groups = Counter(norm(n) for n in names_all)
    ev = [json.loads(l) for l in open(DATA / "eval" / "physics_v1_mech" / "eval_v1.jsonl")]
    nn = set(groups)
    cov = {t: float(np.mean([norm(q["answer"]) in nn for q in ev if q["type"] == t])) for t in ("single", "multihop")}
    gk = 2 * gc.number_of_edges() / gc.number_of_nodes()
    vals, cnts = np.unique(k, return_counts=True)
    ccdf = 1 - np.concatenate([[0], np.cumsum(cnts)[:-1]]) / cnts.sum()
    hubs = [idx[n] for n, _ in sorted(g.degree, key=lambda x: -x[1])[:15]]
    stats = {
        "Entities": f"{n_all:,}", "Relations": f"{m_all:,}",
        "Isolated entities (no links)": f"{len(isolated) / n_all:.0%}",
        "Mean degree": f"{deg_all.mean():.2f}",
        "Connected pieces": f"{len(comps) + len(isolated):,}",
        "Giant component": f"{comps[0]:,} ({comps[0] / n_all:.0%} of entities)",
        "Degree tail exponent α (k≥3)": f"{1 + len(tail) / np.log(tail / 2.5).sum():.2f}",
        "Max degree": f"{deg.max()} (random graph ≈ {int(np.random.default_rng(0).poisson(deg_all.mean(), n_all).max())})",
        "Clustering (giant)": f"{nx.average_clustering(gc):.3f} (random ≈ {gk / gc.number_of_nodes():.5f})",
        "Mean shortest path (giant)": f"{np.mean(dists):.1f} hops (random ≈ {math.log(len(gnames)) / math.log(gk):.1f})",
        "Degree assortativity": f"{nx.degree_assortativity_coefficient(gc):.2f}",
        "Communities (Louvain)": f"{len(set(memb)):,}, modularity Q = {comm.modularity:.2f}",
        "Distinct relation labels": f"{len(lab):,} ({sum(c == 1 for c in lab.values()) / len(lab):.0%} used once)",
        "Facts with ≥2 source passages": f"{(per_edge >= 2).mean():.1%}",
        "Entities with a duplicate name": f"{sum(c for c in groups.values() if c > 1) / n_all:.0%}",
        "CamelCase entity names": f"{np.mean([bool(re.search(r'[a-z][A-Z]', n)) and ' ' not in n for n in names_all]):.0%}",
        "Gold answer is an entity": f"single {cov['single']:.0%}, two-step {cov['multihop']:.0%}",
    }

    types = Counter((g.nodes[n].get("entity_type") or "unknown").lower() for n in names)
    nodes = [[n, (g.nodes[n].get("entity_type") or "unknown").lower(), round(float(pos[i][0]), 1), round(float(pos[i][1]), 1),
              int(deg[i]), first(g.nodes[n].get("description"), 400), " ".join(srcs(g.nodes[n])), memb[i],
              int(n in gcc_set)] for i, n in enumerate(names)]
    edges = [[idx[u], idx[v], (d.get("keywords") or "").replace(",", ", "), first(d.get("description"), 300),
              " ".join(srcs(d))] for u, v, d in g.edges(data=True) if u != v]
    meta = {"source": str(src), "nodes_drawn": len(nodes), "edges": len(edges), "isolated_left_out": len(isolated),
            "types": types.most_common(12), "communities": Counter(memb).most_common(12), "stats": stats,
            "degree_ccdf": [[int(a), float(b)] for a, b in zip(vals, ccdf)], "hubs": hubs}
    json.dump({"meta": meta, "nodes": nodes, "edges": edges}, open(out, "w"), separators=(",", ":"))
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
