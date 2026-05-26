"""
Visual KG explorer for PCM knowledge graphs.

Run:
    python -m ui --kg-dir output/ --conversations sample_data/conversations.json

Then open http://localhost:5050 in your browser.
"""

import json
import re
import argparse
from pathlib import Path

from flask import Flask, jsonify, Response
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef

SCHEMA = Namespace("http://schema.org/")
PROV = Namespace("http://www.w3.org/ns/prov#")
DCTERMS = Namespace("http://purl.org/dc/terms/")
PCM = Namespace("https://w3id.org/pcm/")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

# Properties whose literal values are good candidates for text matching
_MATCHABLE_PROPS = [
    RDFS.label,
    SCHEMA.name,
    SCHEMA.brand,
    SCHEMA.model,
    SCHEMA.color,
    FOAF.name,
    SKOS.prefLabel,
    SKOS.altLabel,
]

_PREFIX_MAP = {
    "http://schema.org/": "schema:",
    "https://w3id.org/ontobio#": "ob:",
    "http://xmlns.com/foaf/0.1/": "foaf:",
    "http://www.w3.org/ns/prov#": "prov:",
    "https://w3id.org/pcm/": "pcm:",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs:",
    "http://purl.org/dc/terms/": "dcterms:",
    "http://www.w3.org/2004/02/skos/core#": "skos:",
    "http://purl.org/vocab/relationship/": "rel:",
    "http://purl.org/vocab/bio/0.1/": "bio:",
    "http://rdfs.org/resume-rdf/cv.rdfs#": "cv:",
    "https://w3id.org/opencare#": "opencare:",
    "http://rdfs.org/sioc/ns#": "sioc:",
}


def load_graph(ttl_path: str) -> Graph:
    g = Graph()
    g.parse(ttl_path, format="turtle")
    return g


def _compact_uri(uri) -> str:
    uri_str = str(uri)
    for full, prefix in _PREFIX_MAP.items():
        if uri_str.startswith(full):
            return prefix + uri_str[len(full):]
    return uri_str

# Types to skip: ontology schema definitions, provenance, conversation structure
_SKIP_TYPES = {
    "http://www.w3.org/2002/07/owl#Class",
    "http://www.w3.org/2002/07/owl#ObjectProperty",
    "http://www.w3.org/2002/07/owl#DatatypeProperty",
    "http://www.w3.org/2002/07/owl#AnnotationProperty",
    "http://www.w3.org/2002/07/owl#SymmetricProperty",
    "http://www.w3.org/2002/07/owl#AsymmetricProperty",
    "http://www.w3.org/2002/07/owl#IrreflexiveProperty",
    "http://www.w3.org/2002/07/owl#TransitiveProperty",
    "http://www.w3.org/2002/07/owl#Ontology",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property",
    "http://schema.org/ActionStatusType",
    "http://rdfs.org/sioc/ns#Post",
    "http://rdfs.org/sioc/ns#Thread",
    "http://rdfs.org/sioc/ns#UserAccount",
    "http://www.w3.org/ns/prov#Entity",
}

app = Flask(__name__)
KG_DIR = None
CONVERSATIONS = None  # list of conversation dicts from conversations.json

# Type -> color mapping for graph nodes
TYPE_COLORS = {
    "Person": "#4f46e5",
    "Vehicle": "#0891b2",
    "Place": "#059669",
    "Organization": "#d97706",
    "Product": "#db2777",
    "Event": "#7c3aed",
    "Action": "#dc2626",
    "BuyAction": "#dc2626",
    "RepairAction": "#ea580c",
    "TravelAction": "#0d9488",
    "Preference": "#8b5cf6",
    "Habit": "#6366f1",
    "PossessedObject": "#0284c7",
    "LifeEvent": "#7c3aed",
    "PurchaseEvent": "#dc2626",
    "ServiceEvent": "#ea580c",
    "SocialEvent": "#059669",
    "TravelEvent": "#0d9488",
    "ProblemEvent": "#b91c1c",
    "Concept": "#64748b",
}

HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PCM Explorer</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; color: #333; }
.container { max-width: 1400px; margin: 0 auto; padding: 20px; }
h1 { margin-bottom: 16px; font-size: 24px; }
.row { display: flex; gap: 16px; margin-bottom: 16px; }
.col { flex: 1; }
label { display: block; font-weight: 600; margin-bottom: 4px; font-size: 13px; }
select, input, textarea, button { font-family: inherit; font-size: 14px; }
select, input { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; }
textarea { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; font-family: 'SF Mono', Monaco, monospace; font-size: 13px; resize: vertical; }
button { padding: 8px 20px; border: none; border-radius: 4px; cursor: pointer; font-weight: 600; }
.btn-primary { background: #2563eb; color: white; }
.btn-primary:hover { background: #1d4ed8; }
.btn-secondary { background: #6b7280; color: white; }
.btn-secondary:hover { background: #4b5563; }
.btn-green { background: #059669; color: white; }
.btn-green:hover { background: #047857; }
.buttons { display: flex; gap: 8px; margin-bottom: 16px; }
.info { background: #dbeafe; padding: 12px; border-radius: 6px; margin-bottom: 16px; font-size: 13px; }
.info strong { color: #1e40af; }
table { width: 100%; border-collapse: collapse; background: white; border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
th { background: #1e293b; color: white; padding: 10px 12px; text-align: left; font-size: 13px; }
td { padding: 8px 12px; border-bottom: 1px solid #e5e7eb; font-size: 13px; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
td:hover { white-space: normal; word-break: break-all; }
tr:hover td { background: #f0f9ff; }
.stats { color: #6b7280; font-size: 13px; margin-top: 8px; }
.error { background: #fef2f2; color: #991b1b; padding: 12px; border-radius: 6px; margin-bottom: 16px; }
.entities { background: white; padding: 12px; border-radius: 6px; font-size: 12px; font-family: monospace; max-height: 200px; overflow-y: auto; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); white-space: pre-wrap; }
#loading { display: none; color: #6b7280; font-style: italic; }
.tab-bar { display: flex; gap: 4px; margin-bottom: 12px; }
.tab { padding: 6px 16px; border-radius: 4px 4px 0 0; cursor: pointer; background: #e5e7eb; font-size: 13px; }
.tab.active { background: white; font-weight: 600; }
.msg { padding: 10px 14px; margin: 6px 0; border-radius: 8px; font-size: 13px; line-height: 1.5; max-width: 85%; }
.msg-user { background: #dbeafe; margin-right: auto; border-bottom-left-radius: 2px; }
.msg-assistant { background: #f1f5f9; margin-left: auto; border-bottom-right-radius: 2px; color: #475569; }
.msg-role { font-size: 11px; font-weight: 600; color: #6b7280; margin-bottom: 2px; }
#graph-container { width: 100%; height: 500px; border: 1px solid #e5e7eb; border-radius: 6px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.graph-legend { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; font-size: 12px; }
.legend-item { display: flex; align-items: center; gap: 4px; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; }
.hl { padding: 1px 3px; border-radius: 3px; cursor: pointer; position: relative; font-weight: 500; }
.hl:hover { filter: brightness(0.9); }
.hl-tip { display: none; position: absolute; bottom: 100%; left: 0; background: #1e293b; color: #e2e8f0; padding: 6px 10px; border-radius: 4px; font-size: 11px; font-weight: 400; white-space: nowrap; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,0.2); pointer-events: none; }
.hl-tip::after { content: ''; position: absolute; top: 100%; left: 12px; border: 5px solid transparent; border-top-color: #1e293b; }
.hl:hover .hl-tip { display: block; }
</style>
</head>
<body>
<div class="container">
<h1>PCM Explorer</h1>

<div class="row">
<div class="col">
  <label>Knowledge Graph</label>
  <select id="kg-select" onchange="loadKG()">
    <option value="">Select a conversation...</option>
  </select>
</div>
</div>

<div id="kg-info" class="info" style="display:none"></div>

<div class="tab-bar">
  <div class="tab active" onclick="switchTab('graph')">Graph</div>
  <div class="tab" onclick="switchTab('entities')">Entities</div>
  <div class="tab" onclick="switchTab('ttl')">TTL</div>
  <div class="tab" onclick="switchTab('messages')">Messages</div>
</div>

<div id="tab-graph">
  <div id="graph-container"></div>
  <div id="graph-legend" class="graph-legend"></div>
  <p id="graph-stats" class="stats"></p>
</div>

<div id="tab-entities" style="display:none">
  <div id="entities" class="entities"></div>
</div>

<div id="tab-ttl" style="display:none">
  <div class="buttons" style="margin-bottom:8px">
    <button class="btn-secondary" onclick="openInProtege()">Open in Protege</button>
  </div>
  <textarea id="ttl-content" rows="25" readonly style="width:100%;font-family:'SF Mono',Monaco,monospace;font-size:12px;background:#1e293b;color:#e2e8f0;padding:12px;border-radius:6px;border:none;resize:vertical"></textarea>
</div>

<div id="tab-messages" style="display:none">
  <div id="messages-content" style="max-height:600px;overflow-y:auto"></div>
</div>

</div>

<script>
let allKGs = [];
let network = null;

async function init() {
  const r = await fetch('/api/kgs');
  allKGs = await r.json();
  const sel = document.getElementById('kg-select');
  allKGs.forEach(kg => {
    const opt = document.createElement('option');
    opt.value = kg.id;
    opt.textContent = kg.description ? `${kg.id} — ${kg.description}` : kg.id;
    sel.appendChild(opt);
  });
}

async function loadKG() {
  const id = document.getElementById('kg-select').value;
  if (!id) return;
  const r = await fetch(`/api/kg/${id}/info`);
  const info = await r.json();
  const div = document.getElementById('kg-info');
  div.style.display = 'block';
  div.innerHTML = `<strong>${esc(id)}</strong> — ${info.triples} triples, ${info.subjects} entities` +
    (info.description ? `<br>${esc(info.description)}` : '');
  document.getElementById('entities').textContent = info.sample;
  loadGraph(id);
  loadTTL(id);
  loadMessages(id);
}

async function loadTTL(id) {
  const r = await fetch(`/api/kg/${id}/ttl`);
  const data = await r.json();
  document.getElementById('ttl-content').value = data.ttl || '';
}

async function loadMessages(id) {
  const r = await fetch(`/api/kg/${id}/annotated-messages`);
  const data = await r.json();
  const div = document.getElementById('messages-content');
  if (!data.messages || data.messages.length === 0) {
    div.innerHTML = '<p class="stats">No messages available.</p>';
    return;
  }
  let html = '';
  data.messages.forEach(m => {
    const cls = m.role === 'user' ? 'msg-user' : 'msg-assistant';
    const body = renderHighlights(m.content, m.spans || []);
    html += `<div class="msg ${cls}"><div class="msg-role">${esc(m.role)}</div>${body}</div>`;
  });
  div.innerHTML = html;
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function renderHighlights(text, spans) {
  if (!spans || spans.length === 0) return esc(text);
  let result = '';
  let pos = 0;
  spans.forEach(sp => {
    if (sp.start > pos) result += esc(text.slice(pos, sp.start));
    const matched = esc(text.slice(sp.start, sp.end));
    const types = (sp.types || []).join(', ');
    const tip = esc(sp.entity_id + ' (' + types + ')');
    const bg = sp.color + '30';  // 30 = ~19% opacity hex
    result += `<span class="hl" style="background:${bg};border-bottom:2px solid ${sp.color}">` +
      `${matched}<span class="hl-tip">${tip}</span></span>`;
    pos = sp.end;
  });
  if (pos < text.length) result += esc(text.slice(pos));
  return result;
}

async function openInProtege() {
  const id = document.getElementById('kg-select').value;
  if (!id) return;
  const r = await fetch(`/api/kg/${id}/open-protege`, {method: 'POST'});
  const data = await r.json();
  if (data.error) alert(data.error);
}

async function loadGraph(id) {
  const r = await fetch(`/api/kg/${id}/graph`);
  const data = await r.json();
  renderGraph(data);
}

function renderGraph(data) {
  const container = document.getElementById('graph-container');
  const nodes = new vis.DataSet(data.nodes.map(n => ({
    id: n.id, label: n.label, color: n.color,
    shape: n.shape || 'dot', size: n.size || 16,
    font: { size: 11, color: '#333' },
    title: `${n.id}\\nTypes: ${n.types||''}`,
  })));
  const edges = new vis.DataSet(data.edges.map(e => ({
    from: e.from, to: e.to, label: e.label,
    arrows: 'to', color: { color: '#94a3b8', highlight: '#2563eb' },
    font: { size: 9, color: '#6b7280', strokeWidth: 0 },
    smooth: { type: 'curvedCW', roundness: 0.15 },
  })));

  const options = {
    physics: { solver: 'forceAtlas2Based', forceAtlas2Based: { gravitationalConstant: -40, springLength: 120 } },
    interaction: { hover: true, tooltipDelay: 100 },
    layout: { improvedLayout: true },
  };

  if (network) network.destroy();
  network = new vis.Network(container, { nodes, edges }, options);

  // Legend
  const legend = document.getElementById('graph-legend');
  const seen = {};
  data.nodes.forEach(n => { if (n.type_label && !seen[n.type_label]) { seen[n.type_label] = n.color; }});
  legend.innerHTML = Object.entries(seen).map(([t,c]) =>
    `<span class="legend-item"><span class="legend-dot" style="background:${c}"></span>${t}</span>`
  ).join('');

  document.getElementById('graph-stats').textContent = `${data.nodes.length} nodes, ${data.edges.length} edges`;
}

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  ['graph','entities','ttl','messages'].forEach(t => {
    document.getElementById('tab-'+t).style.display = t === tab ? 'block' : 'none';
  });
  event.target.classList.add('active');
  if (tab === 'graph' && network) network.fit();
}

init();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


def _safe_ttl_path(kg_id: str):
    """Return the resolved TTL path only if it is inside KG_DIR; else None."""
    resolved = (Path(KG_DIR) / f"{kg_id}.ttl").resolve()
    if not str(resolved).startswith(str(Path(KG_DIR).resolve()) + "/"):
        return None
    return resolved


@app.route("/api/kgs")
def list_kgs():
    kg_path = Path(KG_DIR)
    kgs = []
    for f in sorted(kg_path.glob("*.ttl")):
        kg_id = f.stem
        description = None
        if CONVERSATIONS:
            for conv in CONVERSATIONS:
                if conv["conversation_id"] == kg_id:
                    description = conv.get("description", "")
                    break
        kgs.append({"id": kg_id, "description": description})
    return jsonify(kgs)


@app.route("/api/kg/<kg_id>/info")
def kg_info(kg_id):
    ttl_path = _safe_ttl_path(kg_id)
    if not ttl_path or not ttl_path.exists():
        return jsonify({"error": "Not found"}), 404
    g = load_graph(str(ttl_path))

    # Build a clean entity listing
    lines = []
    seen = set()
    for s in g.subjects(RDF.type, None):
        s_str = str(s)
        if s_str in seen:
            continue
        if not s_str.startswith("https://w3id.org/pcm/"):
            continue
        if "/session/" in s_str or "/msg_" in s_str or "person/agent/" in s_str:
            continue
        entity_types = {str(t) for t in g.objects(s, RDF.type)}
        if entity_types & _SKIP_TYPES:
            continue
        seen.add(s_str)
        types = [_compact_uri(t) for t in g.objects(s, RDF.type)
                 if "owl#" not in str(t)]
        name = None
        for n in g.objects(s, SCHEMA.name):
            name = str(n)
            break
        compact = _compact_uri(s)
        type_str = ", ".join(types)
        if name:
            lines.append(f"- {compact} a {type_str} ; schema:name \"{name}\"")
        else:
            lines.append(f"- {compact} a {type_str}")

    sample = "\n".join(sorted(lines))

    info = {
        "triples": len(g),
        "subjects": len(seen),
        "sample": sample,
    }

    if CONVERSATIONS:
        for conv in CONVERSATIONS:
            if conv["conversation_id"] == kg_id:
                info["description"] = conv.get("description", "")
                break

    return jsonify(info)


@app.route("/api/kg/<kg_id>/graph")
def kg_graph(kg_id):
    """Return nodes and edges for vis-network visualization."""
    ttl_path = _safe_ttl_path(kg_id)
    if not ttl_path or not ttl_path.exists():
        return jsonify({"error": "Not found"}), 404
    g = load_graph(str(ttl_path))

    nodes = {}
    edges = []

    # Collect PCM entities as nodes (skip ontology schema definitions)
    for s in g.subjects():
        s_str = str(s)
        if not s_str.startswith("https://w3id.org/pcm/"):
            continue
        if "/session/" in s_str or "/msg_" in s_str:
            continue

        entity_types = {str(t) for t in g.objects(s, RDF.type)}
        if entity_types & _SKIP_TYPES:
            continue
        if not entity_types:
            continue

        compact = _compact_uri(s)
        name = None
        for n in g.objects(s, URIRef("http://schema.org/name")):
            name = str(n)
            break

        type_labels = []
        for t in g.objects(s, RDF.type):
            tl = _compact_uri(t).split(":")[-1]
            if tl not in ("Resource",):
                type_labels.append(tl)

        color = "#94a3b8"
        primary_type = ""
        for tl in type_labels:
            if tl in TYPE_COLORS:
                color = TYPE_COLORS[tl]
                primary_type = tl
                break

        conn_count = sum(1 for _ in g.predicate_objects(s))
        size = min(8 + conn_count * 2, 30)

        nodes[s_str] = {
            "id": compact,
            "label": name or compact.split(":")[-1],
            "color": color,
            "size": size,
            "shape": "diamond" if "Person" in type_labels else "dot",
            "types": ", ".join(type_labels),
            "type_label": primary_type or (type_labels[0] if type_labels else ""),
        }

    # Collect edges between PCM entities
    for s, p, o in g:
        s_str = str(s)
        o_str = str(o)
        p_str = str(p)

        if s_str not in nodes or o_str not in nodes:
            continue
        if any(p_str.startswith(ns) for ns in ("http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                                                 "http://www.w3.org/ns/prov#")):
            continue

        edge_label = _compact_uri(p).split(":")[-1]
        edges.append({
            "from": nodes[s_str]["id"],
            "to": nodes[o_str]["id"],
            "label": edge_label,
        })

    # Filter out disconnected nodes (no edges) from graph view
    connected_ids = set()
    for e in edges:
        connected_ids.add(e["from"])
        connected_ids.add(e["to"])
    graph_nodes = [n for n in nodes.values() if n["id"] in connected_ids]

    return jsonify({
        "nodes": graph_nodes,
        "edges": edges,
    })


@app.route("/api/kg/<kg_id>/ttl")
def kg_ttl(kg_id):
    """Return the raw TTL file content."""
    ttl_path = _safe_ttl_path(kg_id)
    if not ttl_path or not ttl_path.exists():
        return jsonify({"error": "Not found"}), 404
    return jsonify({"ttl": ttl_path.read_text(encoding="utf-8")})


@app.route("/api/kg/<kg_id>/messages")
def kg_messages(kg_id):
    """Return the conversation messages for this KG."""
    if not CONVERSATIONS:
        return jsonify({"messages": []})
    for conv in CONVERSATIONS:
        if conv["conversation_id"] == kg_id:
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in conv.get("messages", [])
            ]
            return jsonify({"messages": messages})
    return jsonify({"messages": []})


@app.route("/api/kg/<kg_id>/annotated-messages")
def kg_annotated_messages(kg_id):
    """Return messages with highlight spans for text that has associated triples."""
    if not CONVERSATIONS:
        return jsonify({"messages": []})

    conv = None
    for c in CONVERSATIONS:
        if c["conversation_id"] == kg_id:
            conv = c
            break
    if not conv:
        return jsonify({"messages": []})

    ttl_path = _safe_ttl_path(kg_id)
    if not ttl_path or not ttl_path.exists():
        # Fall back to plain messages if no KG
        return jsonify({"messages": [
            {"role": m["role"], "content": m["content"], "spans": []}
            for m in conv.get("messages", [])
        ]})

    g = load_graph(str(ttl_path))

    # Build a map: session_uri -> list of entity info dicts
    session_entities = {}  # str(session_uri) -> [{ id, label, types, color, strings }]
    for s in g.subjects(PROV.wasDerivedFrom, None):
        s_str = str(s)
        if not s_str.startswith("https://w3id.org/pcm/"):
            continue
        if "/session/" in s_str or "/msg_" in s_str:
            continue

        entity_types = {str(t) for t in g.objects(s, RDF.type)}
        if entity_types & _SKIP_TYPES:
            continue

        # Collect matchable strings
        strings = set()
        for prop in _MATCHABLE_PROPS:
            for val in g.objects(s, prop):
                if isinstance(val, Literal):
                    txt = str(val).strip()
                    if len(txt) >= 3:
                        strings.add(txt)

        if not strings:
            continue

        # Determine type label and color
        type_labels = []
        for t in g.objects(s, RDF.type):
            tl = _compact_uri(t).split(":")[-1]
            if tl not in ("Resource",) and "owl#" not in str(t):
                type_labels.append(tl)

        color = "#94a3b8"
        for tl in type_labels:
            if tl in TYPE_COLORS:
                color = TYPE_COLORS[tl]
                break

        compact = _compact_uri(s)
        label = None
        for n in g.objects(s, SCHEMA.name):
            label = str(n)
            break
        if not label:
            for n in g.objects(s, RDFS.label):
                label = str(n)
                break

        entity_info = {
            "id": compact,
            "label": label or compact.split(":")[-1],
            "types": type_labels,
            "color": color,
            "strings": list(strings),
        }

        # Link to each session URI this entity was derived from
        for session_uri in g.objects(s, PROV.wasDerivedFrom):
            sess_str = str(session_uri)
            session_entities.setdefault(sess_str, []).append(entity_info)

    # Now build annotated messages
    # Turn indices in the TTL are 1-based sequential (message 0 → turn 1, message 1 → turn 2, etc.)
    result = []
    for i, m in enumerate(conv.get("messages", [])):
        role = m["role"]
        content = m["content"]
        turn_index = i + 1
        session_uri = f"https://w3id.org/pcm/session/{kg_id}_{turn_index}"
        entities = session_entities.get(session_uri, [])

        # Fallback: match by message content against prov:Entity descriptions
        if not entities and role == "user":
            for sess_str, ents in session_entities.items():
                for d in g.objects(URIRef(sess_str), DCTERMS.description):
                    if str(d).strip() == content.strip():
                        entities = ents
                        break
                if entities:
                    break

        spans = _find_spans(content, entities) if role == "user" else []
        result.append({"role": role, "content": content, "spans": spans})

    return jsonify({"messages": result})


def _find_spans(text, entities):
    """Find non-overlapping highlight spans by matching entity strings in text."""
    if not entities:
        return []

    # Collect all candidate matches: (start, end, entity_info)
    candidates = []
    text_lower = text.lower()
    for ent in entities:
        for s in ent["strings"]:
            s_lower = s.lower()
            # Find all occurrences
            start = 0
            while True:
                idx = text_lower.find(s_lower, start)
                if idx == -1:
                    break
                candidates.append((idx, idx + len(s), ent))
                start = idx + 1

    if not candidates:
        return []

    # Sort by length descending (prefer longer matches), then by position
    candidates.sort(key=lambda c: (-(c[1] - c[0]), c[0]))

    # Greedy non-overlapping selection
    taken = []  # list of (start, end, entity_info)
    for start, end, ent in candidates:
        if not any(s < end and e > start for s, e, _ in taken):
            taken.append((start, end, ent))

    # Sort by position for output
    taken.sort(key=lambda c: c[0])

    return [
        {
            "start": s,
            "end": e,
            "entity_id": ent["id"],
            "entity_label": ent["label"],
            "types": ent["types"],
            "color": ent["color"],
        }
        for s, e, ent in taken
    ]


@app.route("/api/kg/<kg_id>/open-protege", methods=["POST"])
def open_protege(kg_id):
    """Open the TTL file in Protege."""
    import subprocess
    ttl_path = _safe_ttl_path(kg_id)
    if not ttl_path or not ttl_path.exists():
        return jsonify({"error": "Not found"}), 404
    abs_path = str(ttl_path)
    try:
        subprocess.Popen(["open", "/Applications/Protégé.app", abs_path])
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": f"Failed to open Protege: {e}"})


def main():
    global KG_DIR, CONVERSATIONS

    parser = argparse.ArgumentParser(description="Visual explorer for PCM knowledge graphs")
    parser.add_argument("--kg-dir", required=True, help="Directory with TTL files")
    parser.add_argument("--conversations", help="Path to conversations.json (for message viewing)")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    KG_DIR = args.kg_dir

    if args.conversations and Path(args.conversations).exists():
        with open(args.conversations) as f:
            CONVERSATIONS = json.load(f)
        print(f"Loaded {len(CONVERSATIONS)} conversations from {args.conversations}")

    print(f"KG dir: {KG_DIR}")
    print(f"Open http://localhost:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
