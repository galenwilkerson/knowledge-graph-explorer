# Knowledge Graph Explorer

**What a small language model builds when you ask it to turn an encyclopedia into a knowledge graph — and what
that graph looks like through the lens of network science.**

**Live demo:** https://galenwilkerson.github.io/knowledge-graph-explorer/

A 7-billion-parameter model (Qwen2.5-7B-Instruct, via [LightRAG](https://github.com/HKUDS/LightRAG)) read 2,917
passages from 373 Wikipedia articles on mechanics — classical, quantum, relativity, fluids, solids, statistical
mechanics — and extracted entities and the relations between them. Every entity and relation keeps the passages
it came from. This repository contains an interactive viewer for the resulting graph, two notebooks, and the data.

## The interactive viewer (`docs/`)

- Zoom and pan over 22,552 linked entities and 29,551 relations, laid out with ForceAtlas2 and coloured by
  community (named automatically by their best-connected members) or by entity type.
- Hover for details; click an entity to see all its relations with their labels, facts and source passages.
- **Query the graph:** an entity's neighbourhood (`Heisenberg 2`), the shortest chains of facts between two
  entities (`path Galileo -> Bohr`), or patterns (`?:person -[discover]- ?`).
- Filters for entity types, communities, relation types, degree, corroboration, and the giant component.
- Panel of network-science and knowledge-graph statistics and plots, with plain-language explanations.

## What the graph shows

| | |
|---|---|
| Entities / relations | 43,999 / 29,551 |
| Entities with no links at all | 49% |
| Giant connected component | 42% of entities |
| Degree distribution | heavy-tailed, tail exponent α ≈ 2.2; max degree 229 (random graph ≈ 9) |
| Small world (giant component) | mean path 5.7 hops (random ≈ 9.3); clustering 357× random |
| Robustness | removing the top 1% of hubs shrinks the giant component to 63% (random 1%: 98%) |
| Communities | modularity Q = 0.82; large communities match sub-fields of mechanics |
| Relation vocabulary | 10,627 distinct labels, 67% used only once |
| Duplicate names | 26% of entities share a normalised name with another (`WaveFunction` / `wave_function` / ...) |
| Corroboration | 3.5% of relations are supported by more than one passage |

In short: a small model **names** things readily but **connects** them poorly. The connected part is a
topical small world held together by a few generic hubs, and the relation labels form no consistent vocabulary.
These are the limits that matter when such a graph is used for multi-step question answering.

## Notebooks (`notebooks/`)

- **`knowledge_graphs_tutorial.ipynb`** — an introduction: facts as triples, graph queries, schemas, extracting
  a graph from text with provenance, entity resolution and how a careless merge creates false facts, multi-hop
  question answering, and personalized PageRank (the retrieval idea behind HippoRAG). Its saved outputs for
  the final section come from an earlier 200-passage trial graph; re-running it uses the full graph in `data/`.
- **`kg_explore.ipynb`** — a tour of the full graph: snapshots, a structured query and a question traced end to
  end through graph retrieval, standard network-science measures (degree distribution, clustering, path
  lengths, assortativity, k-cores, centralities, communities, robustness), and knowledge-graph quality
  measures (types and schema, relation vocabulary, provenance, redundancy, completeness), with a comparison to
  the HippoRAG graph built from the same passages.

## Data (`data/`)

| File | Contents |
|---|---|
| `graphs/lightrag_main/graph_chunk_entity_relation.graphml` | the LightRAG graph (entities, relations, descriptions, source passage IDs) |
| `graphs/hipporag_main/openie_results_ner_qwen2.5-7b.json` | HippoRAG's open-IE triples from the same passages |
| `corpus/physics_v1_mech/chunks.jsonl` | the 2,917 passages (Wikipedia text, ≤512 tokens each) |
| `eval/physics_v1_mech/eval_v1.jsonl` | 179 questions with gold answers and supporting passages (single-fact, two-step, unanswerable) |

## Reproduce

Python 3.12 with `networkx`, `python-igraph`, `numpy`, `torch` (CPU is enough), `matplotlib`, `jupyter`.

```
python scripts/kg_viz.py data/graphs/lightrag_main docs/lightrag_main_fa2.json      # layout + statistics (~25 min on 14 CPU threads)
python scripts/kg_viz_plots.py data/graphs/lightrag_main docs/lightrag_main_plots.json
python scripts/make_kg_explore.py notebooks/kg_explore.ipynb                          # builds and runs the notebook
python -m http.server --directory docs 8000                                           # then open http://localhost:8000
```

This is part of an ongoing study comparing knowledge-graph retrieval (LightRAG, HippoRAG) with plain retrieval for
small language models on traceable, multi-step questions; the comparison results are not included yet.

## Credits and licences

By Galen Wilkerson — [Komplex AI](https://komplexai.io), 2026.

- **Code** (`scripts/`, `docs/index.html`, notebook code): MIT licence, see `LICENSE`.
- **Data and data-derived content** (`data/`, `docs/*.json`, notebook outputs): derived from
  [Wikipedia](https://en.wikipedia.org/) text, licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/);
  this derived data is shared under the same licence.
- Built with [LightRAG](https://github.com/HKUDS/LightRAG), [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG),
  [Qwen2.5](https://huggingface.co/Qwen), [sigma.js](https://www.sigmajs.org/), [graphology](https://graphology.github.io/),
  [NetworkX](https://networkx.org/) and [igraph](https://igraph.org/).
