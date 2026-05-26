"""
Shared utilities for knowledge graph extraction.

These functions are used by both the core extractor (mellea-based) and the
baseline extractors (AMR, RITS, SHACL, modular, OntoGPT).
"""

import re
import json
from pathlib import Path

from rdflib import Graph, Literal, URIRef, RDF, Namespace, XSD, OWL
from rdflib.namespace import PROV, DCTERMS, RDFS, FOAF

from .rdf_store import RDFMemoryStore


# ---------------------------------------------------------------------------
# Default PromptBuilder — reads ontology at import time
# ---------------------------------------------------------------------------
from .prompt_builder import PromptBuilder as _PromptBuilder

_DEFAULT_ONTOLOGY = Path(__file__).resolve().parent.parent / "ontology" / "pcm.ttl"
_default_prompt_builder = _PromptBuilder(_DEFAULT_ONTOLOGY)

# Public aliases
TURTLE_PREFIXES = _default_prompt_builder.prefixes


def _build_system_prompt(agent_name: str) -> str:
    """Build the system prompt with the agent name and current year."""
    return _default_prompt_builder.system_prompt(agent_name)


def _extract_turtle(raw: str) -> str:
    """Clean up the LLM response to extract valid Turtle."""
    cleaned = re.sub(r"^```(?:turtle|ttl|sparql|rdf)?\s*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?\s*```$", "", cleaned)
    if "@prefix" not in cleaned:
        cleaned = TURTLE_PREFIXES + "\n" + cleaned
    return cleaned


# ---------------------------------------------------------------------------
# Schema conformance validation
# ---------------------------------------------------------------------------

_schema_sets_cache = None

_SKIP_TYPES = {
    str(OWL.Class), str(OWL.ObjectProperty), str(OWL.DatatypeProperty),
    str(OWL.AnnotationProperty), str(OWL.SymmetricProperty),
    str(OWL.AsymmetricProperty), str(OWL.IrreflexiveProperty),
    str(OWL.TransitiveProperty), str(OWL.Ontology),
    str(RDF.Property),
    str(PROV.Entity),
}

_SKIP_PREDICATES = {
    str(RDF.type), str(RDFS.label), str(RDFS.comment),
    str(RDFS.subClassOf), str(RDFS.subPropertyOf),
    str(RDFS.domain), str(RDFS.range),
    str(OWL.equivalentClass), str(OWL.equivalentProperty),
    str(OWL.inverseOf),
    str(PROV.wasDerivedFrom), str(PROV.wasAttributedTo),
    str(PROV.generatedAtTime),
    str(DCTERMS.identifier), str(DCTERMS.description),
}


def _load_schema_sets():
    """Load and cache valid classes and properties from pcm.ttl + pcm-claude.ttl."""
    global _schema_sets_cache
    if _schema_sets_cache is not None:
        return _schema_sets_cache

    ontology_dir = Path(__file__).resolve().parent.parent / "ontology"
    g = Graph()
    g.parse(str(ontology_dir / "pcm.ttl"), format="turtle")
    claude_ttl = ontology_dir / "pcm-claude.ttl"
    if claude_ttl.exists():
        g.parse(str(claude_ttl), format="turtle")

    classes = set()
    for cls_type in [OWL.Class, RDFS.Class]:
        for s in g.subjects(RDF.type, cls_type):
            classes.add(str(s))
    for s in g.subjects(RDFS.subClassOf, None):
        classes.add(str(s))
    for o in g.objects(None, RDFS.subClassOf):
        classes.add(str(o))

    properties = set()
    for prop_type in [OWL.ObjectProperty, OWL.DatatypeProperty,
                      OWL.AnnotationProperty, RDF.Property,
                      OWL.SymmetricProperty, OWL.AsymmetricProperty,
                      OWL.IrreflexiveProperty, OWL.TransitiveProperty]:
        for s in g.subjects(RDF.type, prop_type):
            properties.add(str(s))

    _schema_sets_cache = (classes, properties)
    return _schema_sets_cache


def _strip_nonconforming_triples(g: Graph) -> int:
    """Remove triples with types or properties not in the PCM ontology.

    Returns the number of triples removed.
    """
    valid_classes, valid_properties = _load_schema_sets()
    pcm_ns = "https://w3id.org/pcm/"
    to_remove = []

    for s in set(g.subjects()):
        if not str(s).startswith(pcm_ns):
            continue
        for p, o in g.predicate_objects(s):
            pred_uri = str(p)
            if pred_uri == str(RDF.type):
                type_uri = str(o)
                if type_uri not in _SKIP_TYPES and type_uri not in valid_classes:
                    to_remove.append((s, p, o))
            elif pred_uri not in _SKIP_PREDICATES and pred_uri not in valid_properties:
                to_remove.append((s, p, o))

    for triple in to_remove:
        g.remove(triple)

    if to_remove:
        bad_types = {str(o).split("/")[-1].split("#")[-1]
                     for s, p, o in to_remove if str(p) == str(RDF.type)}
        bad_props = {str(p).split("/")[-1].split("#")[-1]
                     for s, p, o in to_remove if str(p) != str(RDF.type)}
        parts = []
        if bad_types:
            parts.append(f"types: {', '.join(sorted(bad_types))}")
        if bad_props:
            parts.append(f"properties: {', '.join(sorted(bad_props))}")
        print(f"  Stripped {len(to_remove)} non-conforming triples ({'; '.join(parts)})")

    return len(to_remove)


# ---------------------------------------------------------------------------
# Store integration with provenance
# ---------------------------------------------------------------------------

def add_turtle_to_store(
    turtle_str: str,
    store: RDFMemoryStore,
    source_id: str = None,
    source_text: str = None,
    sources: list = None,
    save: bool = True,
    skip_validation: bool = False,
) -> int:
    """
    Parse a Turtle string and merge the resulting triples into the RDF store.

    Handles provenance mapping: replaces pcm:msg_N placeholders with real
    pcm:session/{source_id} URIs and creates prov:Entity nodes for each
    source message.

    Returns:
        Number of triples added
    """
    PCM = Namespace("https://w3id.org/pcm/")

    before = len(store.graph)

    temp = Graph()
    try:
        temp.parse(data=turtle_str, format="turtle")
    except Exception as e:
        print(f"  Turtle parse error: {e}")
        return 0

    if not skip_validation:
        _strip_nonconforming_triples(temp)

    # Normalise into a list of (id, text) pairs
    if sources is None and source_id:
        sources = [(source_id, source_text)]

    if sources:
        msg_to_source = {}
        for i, (sid, stxt) in enumerate(sources):
            msg_uri = PCM[f"msg_{i + 1}"]
            source_uri = PCM[f"session/{sid}"]
            msg_to_source[msg_uri] = (source_uri, sid, stxt)

        has_msg_provenance = any(
            o in msg_to_source
            for _, _, o in temp.triples((None, PROV.wasDerivedFrom, None))
        )

        if has_msg_provenance:
            for s, p, o in list(temp.triples((None, PROV.wasDerivedFrom, None))):
                if o in msg_to_source:
                    source_uri, sid, stxt = msg_to_source[o]
                    temp.remove((s, p, o))
                    temp.add((s, PROV.wasDerivedFrom, source_uri))

        for msg_uri in msg_to_source:
            for p, o in list(temp.predicate_objects(msg_uri)):
                temp.remove((msg_uri, p, o))

        if not has_msg_provenance:
            subjects = set(temp.subjects())
            for sid, stxt in sources:
                source_uri = PCM[f"session/{sid}"]
                for s in subjects:
                    temp.add((s, PROV.wasDerivedFrom, source_uri))

        for source_uri, sid, stxt in msg_to_source.values():
            if (source_uri, RDF.type, PROV.Entity) not in store.graph:
                store.graph.add((source_uri, RDF.type, PROV.Entity))
                store.graph.add((source_uri, DCTERMS.identifier, Literal(sid)))
            if stxt and (source_uri, DCTERMS.description, Literal(stxt)) not in store.graph:
                store.graph.add((source_uri, DCTERMS.description, Literal(stxt)))

    store.graph += temp
    if save:
        store._save()

    return len(store.graph) - before


# ---------------------------------------------------------------------------
# SIOC conversation structure
# ---------------------------------------------------------------------------

def _name_to_uri_local(name: str) -> str:
    """Convert a display name to a URI-safe local name.

    Matches the LLM's convention for person URIs so that SIOC-created
    and LLM-extracted person nodes naturally merge on the same URI.

    E.g. "Alice Smith" -> "Alice_Smith"
         "Jean-Pierre Dupont" -> "Jean-Pierre_Dupont"
    """
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^\w\-]", "", name)
    return name


def _build_person_from_profile(
    store: RDFMemoryStore,
    profile: dict,
) -> "URIRef":
    """Create or update a foaf:Person from a platform profile dict.

    Returns the person URI.
    """
    PCM = Namespace("https://w3id.org/pcm/")
    SCHEMA = Namespace("http://schema.org/")

    display_name = profile.get("display_name", "unknown")
    person_local = _name_to_uri_local(display_name)
    person_uri = PCM[person_local]

    store.graph.add((person_uri, RDF.type, FOAF.Person))
    store.graph.add((person_uri, RDFS.label, Literal(display_name, lang="en")))

    if profile.get("real_name"):
        store.graph.add((person_uri, FOAF.name, Literal(profile["real_name"])))
    if profile.get("email"):
        store.graph.add((
            person_uri, FOAF.mbox,
            Literal(f"mailto:{profile['email']}", datatype=XSD.anyURI),
        ))
    if profile.get("title"):
        store.graph.add((person_uri, SCHEMA.jobTitle, Literal(profile["title"])))
    if profile.get("phone"):
        store.graph.add((person_uri, SCHEMA.telephone, Literal(profile["phone"])))
    if profile.get("timezone"):
        store.graph.add((person_uri, PCM.timezone, Literal(profile["timezone"])))
    if profile.get("image_url"):
        store.graph.add((person_uri, FOAF.img, URIRef(profile["image_url"])))
    if profile.get("slack_user_id"):
        store.graph.add((
            person_uri, SCHEMA.identifier,
            Literal(profile["slack_user_id"]),
        ))

    return person_uri


def _build_sioc_structure(
    store: RDFMemoryStore,
    conversation_id: str,
    messages: list,
    agent_name: str,
    source_platform: str = None,
    channel_type: str = None,
    participants: dict = None,
):
    """
    Build SIOC conversation structure in the store.

    Creates a sioc:Thread for the conversation and a sioc:Post (dual-typed
    as prov:Entity) for each message, with content, role, turn index, and
    reply threading.

    When *participants* is provided (enriched format from Slack/other loaders),
    creates per-sender sioc:UserAccount instances linked to foaf:Person nodes
    with profile data. Otherwise falls back to the legacy two-account model.
    """
    PCM = Namespace("https://w3id.org/pcm/")
    SIOC = Namespace("http://rdfs.org/sioc/ns#")
    SCHEMA = Namespace("http://schema.org/")

    # --- Thread ---
    thread_uri = PCM[f"thread/{conversation_id}"]
    store.graph.add((thread_uri, RDF.type, SIOC.Thread))
    store.graph.add((thread_uri, RDFS.label, Literal(f"Thread {conversation_id}", lang="en")))
    if source_platform:
        store.graph.add((thread_uri, PCM.sourcePlatform, Literal(source_platform)))

    # --- Participant accounts ---
    has_sender_metadata = any(msg.get("sender_id") for msg in messages)

    if has_sender_metadata and participants:
        # Enriched path: per-sender accounts linked to foaf:Person
        account_map = {}  # sender_id -> account_uri
        platform = source_platform or "unknown"

        for sender_id, profile in participants.items():
            if profile.get("is_bot"):
                # Bots get foaf:Agent, not foaf:Person
                bot_name = profile.get("display_name", sender_id)
                person_uri = PCM[_name_to_uri_local(f"Bot_{bot_name}")]
                store.graph.add((person_uri, RDF.type, FOAF.Agent))
                store.graph.add((person_uri, RDFS.label, Literal(bot_name, lang="en")))
            else:
                person_uri = _build_person_from_profile(store, profile)

            account_uri = PCM[f"account/{platform}_{sender_id}"]
            store.graph.add((account_uri, RDF.type, SIOC.UserAccount))
            store.graph.add((account_uri, SIOC.account_of, person_uri))
            store.graph.add((
                account_uri, RDFS.label,
                Literal(profile.get("display_name", sender_id), lang="en"),
            ))
            store.graph.add((account_uri, SCHEMA.identifier, Literal(sender_id)))
            account_map[sender_id] = account_uri
    else:
        # Legacy path: two accounts (user + assistant)
        account_map = None
        agent_id = agent_name.lower().replace(" ", "_")
        user_account = PCM[f"account/{agent_id}"]
        assistant_account = PCM["account/assistant"]
        store.graph.add((user_account, RDF.type, SIOC.UserAccount))
        store.graph.add((user_account, SIOC.account_of, PCM[agent_id]))
        store.graph.add((user_account, RDFS.label, Literal(agent_name, lang="en")))
        store.graph.add((assistant_account, RDF.type, SIOC.UserAccount))
        store.graph.add((assistant_account, RDFS.label, Literal("Assistant", lang="en")))

    # --- Posts ---
    prev_post_uri = None
    for turn_idx, msg in enumerate(messages, 1):
        role = msg["role"]
        text = msg["content"]
        source_id = f"{conversation_id}_{turn_idx}"
        post_uri = PCM[f"session/{source_id}"]

        store.graph.add((post_uri, RDF.type, SIOC.Post))
        store.graph.add((post_uri, RDF.type, PROV.Entity))
        store.graph.add((post_uri, SIOC.has_container, thread_uri))
        store.graph.add((post_uri, SIOC.content, Literal(text)))
        store.graph.add((post_uri, PCM.role, Literal(role)))
        store.graph.add((post_uri, PCM.turnIndex, Literal(turn_idx, datatype=XSD.integer)))
        store.graph.add((post_uri, DCTERMS.identifier, Literal(source_id)))
        store.graph.add((post_uri, DCTERMS.description, Literal(text)))

        # Creator
        if account_map and msg.get("sender_id"):
            creator = account_map.get(msg["sender_id"])
            if creator:
                store.graph.add((post_uri, SIOC.has_creator, creator))
        else:
            creator = user_account if role == "user" else assistant_account
            store.graph.add((post_uri, SIOC.has_creator, creator))

        # Timestamp
        if msg.get("timestamp"):
            store.graph.add((
                post_uri, DCTERMS.created,
                Literal(msg["timestamp"], datatype=XSD.dateTime),
            ))

        # Addressed-to: explicit @mentions
        if account_map:
            for mentioned_id in msg.get("mentioned_users", []):
                target = account_map.get(mentioned_id)
                if target:
                    store.graph.add((post_uri, SIOC.addressed_to, target))
            # DMs: implicitly addressed to the other participant
            if channel_type == "dm" and len(account_map) == 2:
                sender_id = msg.get("sender_id")
                for other_id, other_account in account_map.items():
                    if other_id != sender_id:
                        store.graph.add((
                            post_uri, SIOC.addressed_to, other_account,
                        ))

        # Threading
        if prev_post_uri is not None:
            store.graph.add((prev_post_uri, SIOC.has_reply, post_uri))
        prev_post_uri = post_uri


# ---------------------------------------------------------------------------
# Ontology schema loading
# ---------------------------------------------------------------------------

_ontology_graph_cache = None


def _load_ontology_schema(store: RDFMemoryStore) -> int:
    """
    Merge the ontology schema into the store so rdfs:subClassOf
    relationships are available for querying.
    """
    global _ontology_graph_cache

    if _ontology_graph_cache is None:
        ontology_dir = Path(__file__).resolve().parent.parent / "ontology"
        schema_path = ontology_dir / "pcm.ttl"
        if not schema_path.exists():
            print(f"  Warning: ontology schema not found at {schema_path}")
            return 0
        _ontology_graph_cache = Graph()
        _ontology_graph_cache.parse(str(schema_path), format="turtle")
        claude_ttl = ontology_dir / "pcm-claude.ttl"
        if claude_ttl.exists():
            _ontology_graph_cache.parse(str(claude_ttl), format="turtle")

    before = len(store.graph)
    store.graph += _ontology_graph_cache
    return len(store.graph) - before


# ---------------------------------------------------------------------------
# Conversation loading
# ---------------------------------------------------------------------------

def load_conversations(path: str) -> list:
    """
    Load conversations from PCM sample_data format.

    Expected format:
        [
          {
            "conversation_id": "sample_01",
            "description": "...",
            "messages": [
              {"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."}
            ]
          },
          ...
        ]
    """
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Conversations file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")
