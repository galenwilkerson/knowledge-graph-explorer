"""Plot data for the viewer's statistics panel (viz/index.html loads <name>_plots.json if present). CPU only.

Usage (kg env):  python kg_viz_plots.py <lightrag_dir> <out_json>

Each plot: {"title", "kind": "loglog" | "line" | "bar" | "hbar", "x"/"y" labels, "series": [{"name", "pts": [[x, y], ...]}]
or for bars "bars": [[label, value], ...], "note"}.
"""
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import igraph as ig
import networkx as nx
import numpy as np

SEP = "<SEP>"
DATA = Path(os.environ.get("KG_DATA", Path(__file__).resolve().parent.parent / "data"))


def srcs(a):
    return [c for c in (a.get("file_path") or "").split(SEP) if c]


def norm(s):
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s.strip())
    s = re.sub(r"[-–—/_]", " ", s.lower())
    s = " ".join(re.sub(r"[^\w\s]", "", s).split())
    return s[:-1] if s.endswith("s") and len(s) > 3 else s


def logbin(x, y, nb=20):
    """Average y in log-spaced bins of x (x > 0)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    edges = np.unique(np.logspace(0, np.log10(x.max() + 1), nb).round())
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.any():
            out.append([round(float(x[m].mean()), 2), round(float(y[m].mean()), 5)])
    return out


def ccdf(values):
    """[[x, P(X >= x)], ...] over the distinct values."""
    v = np.sort(np.asarray(values))
    xs, first = np.unique(v, return_index=True)
    return [[int(x), round(float(1 - i / len(v)), 6)] for x, i in zip(xs, first)]


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    G = nx.read_graphml(src / "graph_chunk_entity_relation.graphml")
    G.remove_edges_from(nx.selfloop_edges(G))
    g = G.subgraph([n for n, d in G.degree() if d > 0]).copy()
    gc = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    P = []

    # ---------- network science
    comps = [len(c) for c in nx.connected_components(G)]
    P.append({"title": "Connected component sizes (CCDF)", "kind": "loglog", "x": "component size s (entities)", "y": "P(S ≥ s)",
              "series": [{"name": "components", "pts": ccdf(comps)}],
              "note": "One giant component plus thousands of fragments; size-1 components are the isolated entities."})

    clus = nx.clustering(gc)
    dk = dict(gc.degree())
    P.append({"title": "Clustering vs degree, C(k)", "kind": "loglog", "x": "degree k", "y": "mean clustering",
              "series": [{"name": "C(k)", "pts": [p for p in logbin([dk[n] for n in gc if dk[n] > 1], [clus[n] for n in gc if dk[n] > 1]) if p[1] > 0]}],
              "note": "Falling C(k) ~ k^-1 is the signature of hierarchical, modular structure: hubs join clusters that aren't linked to each other."})

    knn = nx.average_neighbor_degree(gc)
    P.append({"title": "Neighbour degree vs degree, k_nn(k)", "kind": "loglog", "x": "degree k", "y": "mean neighbour degree",
              "series": [{"name": "k_nn(k)", "pts": logbin([dk[n] for n in gc], [knn[n] for n in gc])}],
              "note": "Falling k_nn(k) means disassortative: hubs connect mainly to low-degree entities (hub-and-spoke)."})

    rng = random.Random(0)
    d = Counter(x for s in rng.sample(list(gc), 200) for x in nx.single_source_shortest_path_length(gc, s).values() if x > 0)
    tot = sum(d.values())
    P.append({"title": "Shortest-path lengths (giant component)", "kind": "bar", "x": "hops", "y": "share of pairs",
              "bars": [[str(k), round(v / tot, 4)] for k, v in sorted(d.items())],
              "note": "Sampled from 200 sources. Typical entities are 4–7 hops apart: a small world."})

    core = Counter(nx.core_number(gc).values())
    P.append({"title": "k-core decomposition", "kind": "bar", "x": "core number k", "y": "entities",
              "bars": [[str(k), v] for k, v in sorted(core.items())],
              "note": "The innermost core is the densely interlinked backbone (general relativity, force, wave function, ...)."})

    names = list(gc); idx = {n: i for i, n in enumerate(names)}
    h = ig.Graph(n=len(names), edges=[(idx[u], idx[v]) for u, v in gc.edges()])
    cs = Counter(h.community_multilevel().membership)
    P.append({"title": "Community sizes, Louvain (CCDF)", "kind": "loglog", "x": "community size s", "y": "P(S ≥ s)",
              "series": [{"name": "communities", "pts": ccdf(list(cs.values()))}],
              "note": "A few large topical communities and many small ones."})

    order_t = list(np.argsort(-np.array(h.degree())))
    fr = np.linspace(0, 0.10, 21)
    def gcc_after(order, f):
        g2 = h.copy(); g2.delete_vertices(order[: int(f * len(names))])
        return max(g2.connected_components().sizes()) / len(names)
    targ = [[round(100 * f, 1), round(gcc_after(order_t, f), 4)] for f in fr]
    rand = [[round(100 * f, 1), round(float(np.mean([gcc_after(list(np.random.default_rng(s).permutation(len(names))), f)
                                                      for s in range(3)])), 4)] for f in fr]
    P.append({"title": "Robustness: removing entities", "kind": "line", "x": "% of entities removed", "y": "giant component (fraction)",
              "series": [{"name": "random", "pts": rand}, {"name": "hubs first", "pts": targ}],
              "note": "Robust to random failure, fragile to losing hubs: a few generic entities hold the graph together."})

    # ---------- knowledge graph
    typ = {n: (a.get("entity_type") or "unknown").lower() for n, a in G.nodes(data=True)}
    tc = Counter(typ.values())
    P.append({"title": "Entity types", "kind": "hbar", "x": "entities", "bars": [[t, c] for t, c in tc.most_common(12)],
              "note": f"{len(tc)} distinct types; 'unknown' = the model gave no type."})

    pairs = Counter(" — ".join(sorted((typ[u], typ[v]))) for u, v in G.edges())
    P.append({"title": "Schema: type pairs connected by relations", "kind": "hbar", "x": "relations",
              "bars": [[p, c] for p, c in pairs.most_common(10)], "note": "The implicit schema the LLM produced."})

    lab = Counter(w.strip().lower() for _, _, a in G.edges(data=True) for w in (a.get("keywords") or "").split(",") if w.strip())
    P.append({"title": "Relation-label frequency (CCDF)", "kind": "loglog", "x": "uses u of a label", "y": "P(U ≥ u)",
              "series": [{"name": "relation labels", "pts": ccdf(list(lab.values()))}],
              "note": f"{len(lab):,} distinct labels, {sum(c == 1 for c in lab.values()) / len(lab):.0%} used once: no fixed relation vocabulary."})

    chunks = [json.loads(l)["chunk_id"] for l in open(DATA / "corpus" / "physics_v1_mech" / "chunks.jsonl")]
    epc = Counter(c for _, _, a in G.edges(data=True) for c in srcs(a))
    y = Counter(min(epc.get(c, 0), 30) for c in chunks)
    P.append({"title": "Facts extracted per passage", "kind": "bar", "x": "relations from one passage (30 = 30+)", "y": "passages",
              "bars": [[str(k), v] for k, v in sorted(y.items())], "note": "Extraction yield of the 7B model per 512-token passage."})

    pe = Counter(min(len(srcs(a)), 5) for _, _, a in G.edges(data=True))
    P.append({"title": "Source passages per fact", "kind": "bar", "x": "passages supporting a relation (5 = 5+)", "y": "relations",
              "bars": [[str(k), v] for k, v in sorted(pe.items())], "note": "Most facts rest on a single passage: little corroboration."})

    pn = Counter(min(len(srcs(a)), 10) for _, a in G.nodes(data=True))
    P.append({"title": "Source passages per entity", "kind": "bar", "x": "passages mentioning an entity (10 = 10+)", "y": "entities",
              "bars": [[str(k), v] for k, v in sorted(pn.items())], "note": "Most entities are extracted once and never merged with later mentions."})

    groups = Counter(norm(n) for n in G)
    dup = Counter(min(c, 5) for c in groups.values())
    P.append({"title": "Name variants per concept", "kind": "bar", "x": "entities sharing one normalised name (5 = 5+)", "y": "concepts",
              "bars": [[str(k), v] for k, v in sorted(dup.items())],
              "note": "e.g. WaveFunction / Wave Function / wave_function / WaveFunctions: one concept split across nodes."})

    ev = [json.loads(l) for l in open(DATA / "eval" / "physics_v1_mech" / "eval_v1.jsonl")]
    nn = set(groups)
    cov = defaultdict(list)
    for q in ev:
        if q["type"] != "unanswerable":
            cov[q["type"]].append(norm(q["answer"]) in nn)
    P.append({"title": "Completeness: gold answer is an entity", "kind": "bar", "x": "question type", "y": "share",
              "bars": [[{"single": "single-fact", "multihop": "two-step"}[t], round(float(np.mean(v)), 3)] for t, v in cov.items()],
              "note": "If the answer isn't an entity, graph-only retrieval cannot return it."})

    HELP = {'Connected component sizes (CCDF)': 'A component is a group of entities that can reach each other by following relations. This shows what fraction of groups are at least a given size. One huge group and many tiny ones means much of the knowledge is cut off from the rest.', 'Clustering vs degree, C(k)': "Clustering asks: are an entity's neighbours also linked to each other? Here it is averaged for entities with k links. If well-connected entities have low clustering, they act as bridges between separate topics.", 'Neighbour degree vs degree, k_nn(k)': 'For entities with k links, how well-connected are their neighbours on average? A falling line means hubs mostly attach to minor entities, like a hub-and-spoke airline network.', 'Shortest-path lengths (giant component)': "How many relation hops separate two entities, for pairs in the main connected group. Few hops between almost any two entities is what network scientists call a 'small world'.", 'k-core decomposition': 'Repeatedly strip away entities with fewer than k links. What survives at high k is the dense core of the graph: the central, heavily interlinked concepts.', 'Community sizes, Louvain (CCDF)': 'Communities are groups of entities linked more to each other than to the rest, found automatically (Louvain method). They usually match topics. This shows what fraction of communities are at least a given size.', 'Robustness: removing entities': "What happens to the main connected group as entities are removed, either at random or the most-connected first. A steep drop for 'hubs first' means a few central entities hold the graph together.", 'Entity types': "The category the language model gave each entity when it built the graph (concept, person, method, ...). 'unknown' means it gave none.", 'Schema: type pairs connected by relations': "Which kinds of entities the relations connect, e.g. concept–person. This is the graph's implicit structure; a curated knowledge graph would define it in advance.", 'Relation-label frequency (CCDF)': 'Each relation carries a label written by the model (uses, causes, ...). This shows how often labels are reused. Most labels appear only once, so there is no consistent vocabulary to query by.', 'Facts extracted per passage': 'How many relations the model extracted from each 512-token passage of text. Passages with few relations are nearly invisible to graph-based search.', 'Source passages per fact': 'How many passages support each relation. A fact found in several places is better corroborated; most here rest on a single passage.', 'Source passages per entity': 'How many passages mention each entity. Entities seen in only one passage usually have few links.', 'Name variants per concept': "How many differently spelled entities collapse into one name after simple clean-up (case, spacing, plurals). Each extra variant splits one concept's facts across several nodes.", 'Completeness: gold answer is an entity': "For the test questions, how often the correct answer exists as an entity in the graph. If it doesn't, a system that answers only from the graph can't find it."}
    for p in P:
        p["help"] = HELP.get(p["title"], "")
    json.dump({"plots": P}, open(out, "w"), separators=(",", ":"), default=lambda o: o.item())   # numpy scalars
    print(f"{len(P)} plots -> {out}")


if __name__ == "__main__":
    sys.exit(main())
