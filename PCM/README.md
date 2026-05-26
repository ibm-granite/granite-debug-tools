

# Granite Memory - PCM

[![PyPI version](https://img.shields.io/pypi/v/granite-io?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/granite-io/)
[![Python versions](https://img.shields.io/pypi/pyversions/granite-io?logo=python&logoColor=white)](https://pypi.org/project/granite-io/)
[![License](https://img.shields.io/github/license/ibm-granite/granite-io?color=green)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/ibm-granite/granite-memory-pcm?style=social)](https://github.com/ibm-granite/granite-memory-pcm/stargazers)

A memory system for your own Personal Conversation with an Agent.



# Personal Conversational Memory (PCM)

As AI agents become a primary interface for digital interaction, managing personal data acquired through conversations is increasingly important
 
The methods in this repository are developped so that your personal AI agents can **accumulate relevant knowledge about you** over time, addressing challenges in structuring, updating, tracing provenance, and using such **knowledge about you** at query/inference time. 

We capture salient information in user conversations (with AI agents), and tranforme the copious conversational traces in meaningful and **concise structured representations** that agents can safely use on your behalf, in a transparent, traceble and deterministic fashion to serve our queries and needs.

---

- [⭐ Features](#-features)
- [⚙️ Setup](#%EF%B8%8F-setup)
- [🚀 Run the Extraction on your Data](#-run-the-extraction-on-your-data)
- [🔍 Visualize the Extraction Results](#-visualize-the-extraction-results)
- [🧠 What Do We Capture](#-what-do-we-capture)
- [📚 Read More](#-read-more)

---

## ⭐ Features

### Available Now

1. extract salient information from your lengthy conversation with agents
      - ℹ️ 👉 [What Information do we Capture](#-what-do-we-capture)
      - 🛠️ 👉 [Run the Extraction on your Data](#-run-the-extraction-on-your-data)
2. visualize the extracted information in a web based interface
      - ℹ️ 👉 [Spin your Data UI](#-visualize-the-extraction-results)

      
### Coming Soon

3. ask question about your conversation, and obtain answers from the captured knowledge






<!-- ## Project structure

```
PCM/
├── knowledge_populator/        # KG extraction pipeline
│   ├── __init__.py             # Public API
│   ├── __main__.py             # CLI entry point
│   ├── extractor.py            # LLM extraction, validation, provenance
│   └── rdf_store.py            # RDF graph persistence + SPARQL
├── ui/                         # Web explorer (graph viz, entity browser)
│   ├── __init__.py
│   ├── __main__.py             # CLI entry point
│   └── explorer.py             # Flask app with vis-network visualization
├── query_generator/            # NL → SPARQL (planned)
├── data_model_refiner/         # Ontology adaptation (planned)
├── ontology/
│   ├── pcm.ttl                 # PCM ontology (Turtle/OWL)
│   ├── ontology.md             # Ontology documentation
│   └── future_directions.md    # Planned extensions
├── sample_data/
│   ├── conversations.json      # 12 sample conversations
│   └── questions_answers.json  # Q&A pairs for evaluation
└── requirements.txt
``` -->

## ⚙️ Setup

```bash
# Clone
git clone <repo-url>
cd PCM

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your API key
# Anthropic is the default when no --model is specified (claude-sonnet-4-5)
export ANTHROPIC_API_KEY="sk-ant-..."
# Only needed if using --model gpt-4o or other OpenAI models
export OPENAI_API_KEY="sk-..."
```

## 🚀 Run the Extraction on your Data

In a nutshell (i) you pass the text of your conversation and (ii) obtain a structured represention of [salient information](#-what-do-we-capture) to be used in downstream tasks or to be [visualized](#-visualize-the-extraction-results).
<!-- 
```
Your Conversations (JSON) -> knowledge_populator -> Knowledge Graph (.ttl)
``` -->

<!-- The [knowledge_populator](knowledge_populator) is prompted with the full [PCM ontology](ontology/pcm.ttl) as reference so it knows exactly which classes and properties to use.  -->

<!-- ### Extract from a single text

```bash
python -m knowledge_populator "I just bought a silver Honda Civic on February 10th." --agent Alice
``` -->

We collected [12 sample conversations](sample_data/conversations.json) to get you started - they follow the standard `{role, content}` format:

```json
[
  {
    "conversation_id": "sample_01",
    "description": "Recipes, health, travel, pets",
    "messages": [
      {"role": "user", "content": "I'm thinking of trying new recipes..."},
      {"role": "assistant", "content": "Here are some suggestions..."}
    ]
  }
]
```

Here's the one line command to process all those conversations and obtain extracted knowledge in a single graph:

```bash
python -m knowledge_populator -f sample_data/conversations.json --batch-size 50 -o output/kg.ttl
```

See [knowledge_populator](knowledge_populator/README.md) for more options, CLI reference, and Python API.

## 🔍 Visualize the Extraction Results

We provide the KG Explorer, a web UI for browsing and visualizing extracted knowledge graphs.

Assuming you have already extracted your structured memory

```bash
# Step 1 — Extract per-conversation KGs (one TTL file each)
python -m knowledge_populator -f sample_data/conversations.json --batch-size 50 --per-conversation -o output/
```
You can use visulaize all extracted knowledge runninng the following command:

```bash
# Step 2 — Launch the explorer
python -m ui --kg-dir output/ --conversations sample_data/conversations.json
```

and then opening http://localhost:5050 in your browser.

#### What you can do

Select a conversation from the sidebar, then explore it through four tabs:

- **Graph** — Interactive node-edge visualization (vis-network), color-coded by entity type. Drag, zoom, and click nodes to inspect.
- **Entities** — Browse all extracted entities with their types and names in a searchable table.
- **TTL** — Raw Turtle viewer with syntax highlighting.
- **Messages** — View the original conversation. User messages that produced triples are highlighted with the extracted entities shown on hover.

See [ui](ui/README.md) for CLI options and details.

## 🧠 What Do We Capture

The core concepts that we currently capture are **general preferences and everyday living** including actions, events, products, places, organizations, personal life (family, travel, habits, food, traits, education), identity and social connections, relationships, health, etc.

We reuse standard vocabularies to formalize such information, and define a "schema" for the covered information, formalized in the [PCM ontology](ontology/pcm.ttl), which reuses standard vocabularies:

| Prefix | Source | Covers |
|--------|--------|--------|
| `schema:` | Schema.org | Actions, events, products, places, organizations |
| `ob:` | OntoBio | Family, travel, habits, food, traits, education |
| `foaf:` | FOAF | Person identity, social connections |
| `prov:` | PROV-O | Provenance (every triple traces to its source message) |
| `rel:` | Relationship vocab | Social relationships (friend, colleague, neighbor) |
| `opencare:` | OpenCare | Health records, symptoms, treatments |
| `sioc:` | SIOC | Conversation structure (threads, posts, turns) |
| `pcm:` | **PCM** | Preferences, pets, clothing, plants, collections, problems, services |

<!-- ## Input format

Conversations follow the standard `{role, content}` format:

```json
[
  {
    "conversation_id": "sample_01",
    "description": "Recipes, health, travel, pets",
    "messages": [
      {"role": "user", "content": "I'm thinking of trying new recipes..."},
      {"role": "assistant", "content": "Here are some suggestions..."}
    ]
  }
]
``` -->

<!-- ## Dependencies

- [rdflib](https://rdflib.readthedocs.io/) — RDF graph processing
- [mellea](https://github.com/generative-computing/mellea) — LLM extraction with validation loops
- [litellm](https://docs.litellm.ai/) — Universal LLM provider interface
- [flask](https://flask.palletsprojects.com/) — Web UI for the KG explorer -->

## 📚 Read More

If you are interesed in how we used PCM so far, you can read more in our papers:

- Sungeun An and Anna Lisa Gentile. **Personal Agents and Conversational Memory**. In *Trust, Autonomy and Accountability in PKG-Based Agentic AI Workshop at ESWC 2026*, CEUR Workshop Proceedings, 2026. [[PDF](TAAPAAI26_paper_6.pdf)]

```bibtex
@inproceedings{An2025,
  author       = {Sungeun An and
                  Anna Lisa Gentile},
  editor       = {John Domingue, Aidan Hogan, Sabrina Kirrane, and Oshani Seneviratne},
  title        = {Personal Agents and Conversational Memory},
  booktitle    = {Trust, Autonomy and Accountability in PKG-Based Agentic AI Workshop at the European Semantic Web Conference 2026},
  series       = {{CEUR} Workshop Proceedings},
  pages        = {to appear},
  publisher    = {CEUR-WS.org},
  year         = {2026},
  url          = {https://ceur-ws.org/Vol}
}
```

More technical details on the individual components:
- [knowledge_populator](knowledge_populator/README.md) — CLI reference, model options, and Python API
- [ui](ui/README.md) — KG Explorer CLI options and details
- [ontology](ontology/ontology.md) — Ontology documentation and gap analysis and the [PCM ontology schema](ontology/pcm.ttl)

