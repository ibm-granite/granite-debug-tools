"""
Build TURTLE_PREFIXES, ONTOLOGY_REFERENCE, and SYSTEM_PROMPT at runtime
by reading a reference OWL/Turtle ontology file (e.g. pcm.ttl).

Truly dynamic: the prompt rules, example, and namespace handling are all
derived from whatever ontology is passed in. No hardcoded PCM assumptions.

Usage:
    from .prompt_builder import PromptBuilder

    pb = PromptBuilder("ontology/pcm.ttl")
    pb.prefixes        # @prefix block for Turtle output
    pb.ontology_ref    # human-readable catalog of types & properties
    pb.system_prompt("Alice")  # fully assembled extraction prompt
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Optional

from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD


# ── well-known namespace metadata ──────────────────────────────────────────
# Maps namespace URI → (preferred prefix, short human label).
# Used for prettier output when a namespace is recognized. Ontologies
# with unknown namespaces still work — the prefix from the source file
# is used, and the label defaults to the prefix.
_KNOWN_NS: dict[str, tuple[str, str]] = {
    "http://schema.org/":                          ("schema",   "Schema.org"),
    "https://schema.org/":                         ("schema",   "Schema.org"),
    "https://w3id.org/ontobio#":                   ("ob",       "OntoBio"),
    "http://xmlns.com/foaf/0.1/":                  ("foaf",     "FOAF"),
    "http://www.w3.org/ns/prov#":                  ("prov",     "PROV-O"),
    "http://www.w3.org/2004/02/skos/core#":        ("skos",     "SKOS"),
    "http://purl.org/vocab/relationship/":         ("rel",      "Relationship vocab"),
    "http://purl.org/vocab/bio/0.1/":              ("bio",      "BIO"),
    "http://rdfs.org/resume-rdf/cv.rdfs#":         ("cv",       "CV/ResumeRDF"),
    "https://w3id.org/opencare#":                  ("opencare", "OpenCare"),
    "http://rdfs.org/sioc/ns#":                    ("sioc",     "SIOC"),
    "https://w3id.org/pcm/":                       ("pcm",      "PCM Custom Extensions"),
    "http://www.w3.org/2001/XMLSchema#":           ("xsd",      "XSD"),
    "http://www.w3.org/2000/01/rdf-schema#":       ("rdfs",     "RDFS"),
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": ("rdf",      "RDF"),
    "http://www.w3.org/2002/07/owl#":              ("owl",      "OWL"),
    "http://purl.org/dc/terms/":                   ("dct",      "Dublin Core Terms"),
    "http://purl.org/dc/elements/1.1/":            ("dc",       "Dublin Core"),
    "http://usefulinc.com/ns/doap#":               ("doap",     "DOAP"),
    "http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#": ("nfo", "NFO"),
    "http://purl.org/net/p-plan#":                 ("p-plan",   "P-Plan"),
}

# Namespace URIs that are infrastructure — skip when listing domain types/properties
_INFRA_NS = {
    str(RDF), str(RDFS), str(OWL), str(XSD),
    "http://purl.org/dc/terms/",
    "http://purl.org/dc/elements/1.1/",
    "http://www.w3.org/2002/07/owl#",
}

# ── generic system prompt template ────────────────────────────────────────
# All placeholders are filled at runtime from the ontology.
_SYSTEM_PROMPT_TEMPLATE = """\
You are a knowledge graph extraction engine.

Given a natural language message from a user named {agent_name}, extract \
structured knowledge as RDF triples in Turtle format.

## Rules
1. Output ONLY valid Turtle (TTL) — no explanation, no markdown fences.
2. Always start with these prefixes:
{prefixes}
3. Use the ontology's namespaces for entity URIs. \
Create URI local names from the SPECIFIC entity name in the source text \
(e.g. onto:AWH_Engineering_College, onto:John_Smith). \
Never use generic descriptions like "The film", "The series", or "The company" \
as entity names — always resolve to the actual name mentioned in the text.
4. Use ONLY types and properties defined in the ontology below. \
Do NOT invent new types or properties.
5. Use the most specific type available from the ontology.
6. Extract dates from text when present. Format as "YYYY-MM-DD"^^xsd:date.
7. Extract ALL meaningful knowledge from the sentence: entities, attributes, \
relationships, events, quantities, dates.
8. Every entity MUST have an rdfs:label matching the specific entity name \
from the source text \
(e.g. rdfs:label "Bleach: Hell Verse", not "The film").
9. NO ISOLATED NODES: every entity must be connected to at least one other \
entity or have at least one data property.
10. String literals use quotes: "value". Numbers: use plain literals.
11. If no extractable knowledge is found, output only the prefix block.
{domain_rules}
## Available Types and Properties
{ontology_reference}

Now extract knowledge from the following message(s). Output ONLY valid Turtle.\
"""

# ── PCM-specific rules (only injected when the ontology contains pcm:) ────
_PCM_RULES = """\

## PCM-Specific Rules
- The primary agent is ALWAYS "{agent_name}". Use URI pcm:{agent_id} for them.
- All resource URIs MUST use the pcm: namespace \
(e.g. pcm:user, pcm:silver-honda-civic).
- Dual-type with schema.org actions and domain-specific classes where applicable \
(e.g. a repair is schema:RepairAction + pcm:ServiceEvent; \
a trip is schema:TravelAction + ob:Travel).
- Every event/action MUST include schema:agent. If the message is in \
first person ("I bought…", "I went…") the agent is pcm:{agent_id}.
- For completed actions use schema:actionStatus schema:CompletedActionStatus. \
For planned/intended actions use schema:actionStatus schema:PotentialActionStatus.
- Every pcm:ProblemEvent MUST include a pcm:affects link. \
Every pcm:ServiceEvent and schema:RepairAction MUST include schema:object.
- RELATIONSHIPS: Use rel: for social relationships, ob: for family relationships.
- HEALTH: Use opencare: for health records.
- TRAVEL: Use ob:Travel with ob:travelFrom, ob:travelTo, ob:modeOfTransport.
- HABITS: Use ob:Habit with schema:Schedule for recurring behaviors.
- SHELL COMMANDS: For every pcm:ShellCommand, always set rdfs:comment to the full command string (the same value as pcm:command). Also set pcm:executable to the first token (the binary/program name, e.g. "git", "python3", "ls") and pcm:arguments to everything after the first token.
"""


class PromptBuilder:
    """Reads an OWL/Turtle ontology and builds extraction prompts at runtime.

    Fully dynamic: works with any ontology, not just PCM.  When pcm.ttl is
    loaded, PCM-specific rules are injected automatically.
    """

    def __init__(self, ontology_path: str | Path) -> None:
        self._path = Path(ontology_path)
        if not self._path.is_file():
            raise FileNotFoundError(f"Ontology not found: {self._path}")

        # Parse prefixes declared in the source file (not rdflib defaults)
        self._prefix_map: dict[str, str] = self._parse_source_prefixes(self._path)

        self._graph = Graph()
        self._graph.parse(str(self._path), format="turtle")

        # Auto-discover namespaces from class/property URIs in the graph
        # that are not covered by any declared prefix
        self._auto_register_namespaces()

        # Always include xsd, rdfs, rdf for Turtle output
        for ns_uri, (pfx, _) in _KNOWN_NS.items():
            if pfx in ("xsd", "rdfs", "rdf"):
                self._prefix_map.setdefault(pfx, ns_uri)

        # Detect if this is a PCM ontology
        self._is_pcm = any(
            ns_uri == "https://w3id.org/pcm/"
            for ns_uri in self._prefix_map.values()
        )

        # Cache built artifacts
        self._prefixes: Optional[str] = None
        self._ontology_ref: Optional[str] = None

    def _auto_register_namespaces(self) -> None:
        """Discover namespaces used by classes/properties in the graph and
        register any that don't already have a prefix.

        This handles ontologies where entity URIs use sub-namespaces not
        declared in @prefix lines (e.g. Text2KGBench uses /concepts# and
        /relations# under the main ontology namespace).
        """
        g = self._graph
        known_ns = set(self._prefix_map.values())

        # Collect all namespaces from class and property URIs
        discovered: set[str] = set()
        for uri_type in [OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty,
                         OWL.AnnotationProperty, RDF.Property]:
            for s in g.subjects(RDF.type, uri_type):
                if isinstance(s, BNode):
                    continue
                ns, local = self._split_uri(str(s))
                if ns and local and ns not in known_ns and ns not in _INFRA_NS:
                    discovered.add(ns)

        if not discovered:
            return

        # Try to assign meaningful short prefixes
        used_pfx = set(self._prefix_map.keys())
        for ns in sorted(discovered):
            # Try to derive a prefix from the URI path
            # e.g. .../ont_1_university/concepts# → "concepts"
            #      .../ont_1_university/relations# → "relations"
            candidate = ns.rstrip("#/").rsplit("/", 1)[-1].lower()
            candidate = re.sub(r"[^a-z0-9]", "", candidate)
            if not candidate or candidate in used_pfx:
                # Fallback: ns0, ns1, ...
                i = 0
                while f"ns{i}" in used_pfx:
                    i += 1
                candidate = f"ns{i}"
            self._prefix_map[candidate] = ns
            used_pfx.add(candidate)

    # ── public properties ──────────────────────────────────────────────────

    @property
    def prefixes(self) -> str:
        """Turtle @prefix block derived from the ontology's namespace declarations."""
        if self._prefixes is None:
            self._prefixes = self._build_prefixes()
        return self._prefixes

    @property
    def ontology_ref(self) -> str:
        """Human-readable catalog of all types and properties in the ontology."""
        if self._ontology_ref is None:
            self._ontology_ref = self._build_ontology_reference()
        return self._ontology_ref

    def system_prompt(self, agent_name: str) -> str:
        """Fully assembled system prompt ready to send to the LLM."""
        agent_id = agent_name.lower().replace(" ", "_")
        current_year = datetime.now().year

        # Build domain-specific rules (PCM-specific or empty)
        if self._is_pcm:
            domain_rules = _PCM_RULES.format(
                agent_name=agent_name,
                agent_id=agent_id,
                current_year=current_year,
            )
        else:
            domain_rules = ""

        return _SYSTEM_PROMPT_TEMPLATE.format(
            agent_name=agent_name,
            agent_id=agent_id,
            current_year=current_year,
            prefixes=self.prefixes,
            ontology_reference=self.ontology_ref,
            domain_rules=domain_rules,
        )

    # ── prefix builder ─────────────────────────────────────────────────────

    def _build_prefixes(self) -> str:
        lines = []
        for pfx in sorted(self._prefix_map):
            ns = self._prefix_map[pfx]
            lines.append(f"@prefix {pfx + ':':<10s} <{ns}> .")
        return "\n".join(lines) + "\n"

    # ── ontology reference builder ─────────────────────────────────────────

    def _build_ontology_reference(self) -> str:
        """Walk the ontology graph and produce a markdown catalog grouped by
        namespace, with sections for classes and properties."""

        g = self._graph

        # Reverse-map: namespace URI → prefix
        # Also handle http/https variants (rdflib may normalize schema.org)
        ns_to_pfx: dict[str, str] = {}
        for pfx, ns_uri in self._prefix_map.items():
            ns_to_pfx[ns_uri] = pfx
            # Add http↔https alias for schema.org and similar
            if ns_uri.startswith("http://"):
                ns_to_pfx[ns_uri.replace("http://", "https://", 1)] = pfx
            elif ns_uri.startswith("https://"):
                ns_to_pfx[ns_uri.replace("https://", "http://", 1)] = pfx

        # Collect classes and properties, grouped by namespace
        classes_by_ns: dict[str, list[tuple[str, str, str, list[str]]]] = defaultdict(list)
        props_by_ns: dict[str, list[tuple[str, str, str, str, str]]] = defaultdict(list)

        # ── Classes ────────────────────────────────────────────────────────
        for cls_uri in sorted(set(g.subjects(RDF.type, OWL.Class))):
            if isinstance(cls_uri, BNode):
                continue
            ns, local = self._split_uri(str(cls_uri))
            if ns in _INFRA_NS or not local:
                continue
            pfx = ns_to_pfx.get(ns)
            if pfx is None:
                continue

            label = self._get_label(cls_uri)
            comment = self._get_comment(cls_uri)

            # Superclasses
            supers = []
            for sc in g.objects(cls_uri, RDFS.subClassOf):
                if isinstance(sc, BNode):
                    continue
                sc_curie = self._to_curie(str(sc), ns_to_pfx)
                if sc_curie:
                    supers.append(sc_curie)

            desc = label or local
            if comment:
                desc = comment
            if supers:
                desc += f" (subclass of {', '.join(supers)})"

            classes_by_ns[ns].append((local, f"{pfx}:{local}", desc, supers))

        # ── Properties ─────────────────────────────────────────────────────
        prop_types = [
            OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty,
            RDF.Property,
        ]
        seen_props: set[str] = set()
        for pt in prop_types:
            for prop_uri in g.subjects(RDF.type, pt):
                if isinstance(prop_uri, BNode):
                    continue
                prop_str = str(prop_uri)
                if prop_str in seen_props:
                    continue
                seen_props.add(prop_str)

                ns, local = self._split_uri(prop_str)
                if ns in _INFRA_NS or not local:
                    continue
                pfx = ns_to_pfx.get(ns)
                if pfx is None:
                    continue

                label = self._get_label(prop_uri)
                comment = self._get_comment(prop_uri)
                desc = comment or label or local

                # Domain and range hints
                domain = self._range_hint(prop_uri, RDFS.domain, ns_to_pfx)
                range_ = self._range_hint(prop_uri, RDFS.range, ns_to_pfx)

                props_by_ns[ns].append((local, f"{pfx}:{local}", desc, domain, range_))

        # ── Assemble markdown ──────────────────────────────────────────────
        sections: list[str] = []

        # Determine section ordering: group by namespace, sorted by human label
        all_ns = sorted(
            set(classes_by_ns.keys()) | set(props_by_ns.keys()),
            key=lambda n: _KNOWN_NS.get(n, (n, n))[1],
        )

        for ns in all_ns:
            pfx = ns_to_pfx.get(ns, "?")
            human = _KNOWN_NS.get(ns, (pfx, pfx))[1]

            cls_items = classes_by_ns.get(ns, [])
            prop_items = props_by_ns.get(ns, [])
            if not cls_items and not prop_items:
                continue

            parts: list[str] = []

            if cls_items:
                parts.append(f"## {human} Types (prefix {pfx}:)")
                for _, curie, desc, _ in sorted(cls_items):
                    parts.append(f"- {curie} — {desc}")

            if prop_items:
                parts.append(f"\n## {human} Properties (prefix {pfx}:)")
                for _, curie, desc, domain, range_ in sorted(prop_items):
                    hint = ""
                    if domain and range_:
                        hint = f" [{domain} → {range_}]"
                    elif range_:
                        hint = f" [→ {range_}]"
                    parts.append(f"- {curie} — {desc}{hint}")

            sections.append("\n".join(parts))

        return "\n\n".join(sections) + "\n"

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_source_prefixes(path: Path) -> dict[str, str]:
        """Read @prefix declarations directly from the Turtle source file,
        avoiding the extra default prefixes that rdflib injects."""
        prefix_map: dict[str, str] = {}
        prefix_re = re.compile(
            r"@prefix\s+(\w*):\s+<([^>]+)>\s*\."
        )
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                m = prefix_re.match(line.strip())
                if m:
                    pfx, ns = m.group(1), m.group(2)
                    if pfx:  # skip empty (default) prefix
                        prefix_map[pfx] = ns
        return prefix_map

    @staticmethod
    def _split_uri(uri: str) -> tuple[str, str]:
        """Split a URI into (namespace, local_name)."""
        for sep in ("#", "/"):
            idx = uri.rfind(sep)
            if idx >= 0:
                return uri[: idx + 1], uri[idx + 1 :]
        return uri, ""

    def _get_label(self, uri: URIRef) -> str:
        for obj in self._graph.objects(uri, RDFS.label):
            return str(obj)
        return ""

    def _get_comment(self, uri: URIRef) -> str:
        for obj in self._graph.objects(uri, RDFS.comment):
            return str(obj)
        return ""

    def _to_curie(self, uri: str, ns_to_pfx: dict[str, str]) -> Optional[str]:
        ns, local = self._split_uri(uri)
        pfx = ns_to_pfx.get(ns)
        if pfx and local:
            return f"{pfx}:{local}"
        return None

    def _range_hint(
        self, prop_uri: URIRef, predicate: URIRef, ns_to_pfx: dict[str, str]
    ) -> str:
        values = []
        for obj in self._graph.objects(prop_uri, predicate):
            if isinstance(obj, BNode):
                continue
            curie = self._to_curie(str(obj), ns_to_pfx)
            if curie:
                values.append(curie)
            else:
                # Handle XSD types
                s = str(obj)
                if "XMLSchema" in s:
                    local = s.rsplit("#", 1)[-1]
                    values.append(f"xsd:{local}")
        return ", ".join(sorted(values))
