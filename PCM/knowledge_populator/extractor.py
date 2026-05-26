"""
Extract a personal knowledge graph from conversational text using an LLM.

Uses mellea (generative-computing/mellea) with the litellm backend to call
any LLM provider (Anthropic, OpenAI, Ollama, etc.).
The LLM produces Turtle (TTL) directly, which is parsed by rdflib and
merged into the RDFMemoryStore.
"""

import os

from rdflib import Graph

from .rdf_store import RDFMemoryStore

try:
    from mellea import start_session
    from mellea.core import Requirement, ValidationResult
    from mellea.stdlib.sampling import RepairTemplateStrategy
except ImportError:
    raise ImportError(
        "mellea is required. Install with: pip install mellea"
    )

# ---------------------------------------------------------------------------
# Import shared utilities from utils.py
# ---------------------------------------------------------------------------
from .utils import (
    _default_prompt_builder,
    TURTLE_PREFIXES,
    _build_system_prompt,
    _extract_turtle,
    _strip_nonconforming_triples,
    _load_schema_sets,
    _load_ontology_schema,
    _build_sioc_structure,
    add_turtle_to_store,
    load_conversations,
)


# ---------------------------------------------------------------------------
# Core extraction helpers
# ---------------------------------------------------------------------------

def _validate_turtle(context) -> "ValidationResult":
    """Validate that the LLM output is parseable Turtle."""
    raw = str(context.last_output())
    cleaned = _extract_turtle(raw)

    g = Graph()
    try:
        g.parse(data=cleaned, format="turtle")
    except Exception as e:
        return ValidationResult(False, reason=f"Invalid Turtle syntax: {e}")

    if len(g) == 0:
        return ValidationResult(False, reason="Parsed successfully but graph contains no triples")

    return ValidationResult(True)


def _build_requirements() -> list:
    """Build the list of mellea Requirements for Turtle extraction."""
    return [
        Requirement(
            description="Output must be valid Turtle (TTL) parseable by rdflib",
            validation_fn=_validate_turtle,
        ),
    ]


def _create_session(model: str = "anthropic/claude-sonnet-4-5-20250929"):
    """Create a mellea session with the litellm backend."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        if auth_token:
            os.environ["ANTHROPIC_API_KEY"] = auth_token

    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    return start_session(
        backend_name="litellm",
        model_id=model,
        base_url=base_url,
    )


# ---------------------------------------------------------------------------
# Single / batch extraction
# ---------------------------------------------------------------------------

def extract_turtle_from_text(
    text: str,
    agent_name: str = "User",
    model: str = "anthropic/claude-sonnet-4-5-20250929",
    temperature: float = 0.0,
    session=None,
    prompt_builder=None,
) -> str:
    """
    Extract a knowledge graph from a single text message.

    Args:
        prompt_builder: Optional PromptBuilder instance. When provided,
            uses its dynamically generated prompt instead of the hardcoded one.

    Returns:
        Valid Turtle string with extracted triples
    """
    pb = prompt_builder or _default_prompt_builder
    system_prompt = pb.system_prompt(agent_name)
    prefixes = pb.prefixes
    requirements = _build_requirements()

    try:
        m = session or _create_session(model)
        result = m.instruct(
            f"{system_prompt}\n\nMessage:\n{text}",
            requirements=requirements,
            strategy=RepairTemplateStrategy(loop_budget=3),
            model_options={"temperature": temperature, "max_tokens": 8192},
        )
        raw = str(result)
        m.reset()
    except Exception as e:
        print(f"  LLM error: {e}")
        return prefixes

    return _extract_turtle(raw)


def extract_turtle_from_texts(
    texts: list,
    agent_name: str = "User",
    model: str = "anthropic/claude-sonnet-4-5-20250929",
    temperature: float = 0.0,
    session=None,
    prompt_builder=None,
) -> str:
    """
    Extract a knowledge graph from multiple texts in a single LLM call.
    """
    pb = prompt_builder or _default_prompt_builder
    prefixes = pb.prefixes

    if not texts:
        return prefixes

    if len(texts) == 1:
        return extract_turtle_from_text(
            texts[0], agent_name=agent_name, model=model,
            temperature=temperature, session=session,
            prompt_builder=prompt_builder,
        )

    system_prompt = pb.system_prompt(agent_name)
    requirements = _build_requirements()

    messages_block = "\n\n".join(
        f"Message {i + 1}:\n{text}" for i, text in enumerate(texts)
    )

    provenance_instruction = (
        "\n\nIMPORTANT — Provenance: For every entity or action you extract, "
        "add one or more prov:wasDerivedFrom triples linking it to the "
        "specific message(s) it was extracted from, using pcm:msg_1, "
        "pcm:msg_2, … pcm:msg_N (matching the Message numbers above). "
        "Only link an entity to the message(s) where it is explicitly "
        "mentioned — do NOT link it to all messages."
    )

    try:
        m = session or _create_session(model)
        result = m.instruct(
            f"{system_prompt}{provenance_instruction}\n\n{messages_block}",
            requirements=requirements,
            strategy=RepairTemplateStrategy(loop_budget=3),
            model_options={
                "temperature": temperature,
                "max_tokens": min(8192 * len(texts), 64000),
            },
        )
        raw = str(result)
        m.reset()
    except Exception as e:
        print(f"  LLM error: {e}")
        return prefixes

    return _extract_turtle(raw)


# ---------------------------------------------------------------------------
# Conversation processing
# ---------------------------------------------------------------------------

def extract_conversation(
    conversation: dict,
    store: RDFMemoryStore,
    agent_name: str = "User",
    model: str = "anthropic/claude-sonnet-4-5-20250929",
    batch_size: int = 1,
    session=None,
    prompt_builder=None,
    es_retriever=None,
    user_id: str = None,
) -> int:
    """
    Process a single conversation through the extraction pipeline.

    Builds SIOC structure for all messages (user + assistant), then
    extracts KG triples from user messages only. When es_retriever and
    user_id are provided, indexes every message into Elasticsearch in
    parallel with the KG so both stores stay in sync.

    Args:
        es_retriever: Optional ESRetriever instance for parallel indexing
        user_id:      Memory owner identity for the ES index

    Returns:
        Number of triples added
    """
    conv_id = conversation["conversation_id"]
    messages = conversation["messages"]

    _build_sioc_structure(
        store, conv_id, messages, agent_name,
        source_platform=conversation.get("source_platform"),
        channel_type=conversation.get("channel_type"),
        participants=conversation.get("participants"),
    )

    # Index every message (user + assistant) into ES alongside the KG
    if es_retriever is not None and user_id is not None:
        for turn_idx, msg in enumerate(messages, 1):
            es_retriever.index_message(
                user_id=user_id,
                session_id=conv_id,
                message_id=f"{conv_id}_{turn_idx}",
                role=msg["role"],
                content=msg["content"],
            )

    user_messages = []
    for turn_idx, msg in enumerate(messages, 1):
        if msg["role"] == "user":
            source_id = f"{conv_id}_{turn_idx}"
            user_messages.append((source_id, msg["content"]))

    if not user_messages:
        return 0

    chunks = [
        user_messages[i : i + batch_size]
        for i in range(0, len(user_messages), batch_size)
    ]

    total_added = 0
    for chunk in chunks:
        texts = [text for _, text in chunk]
        sources = [(sid, text) for sid, text in chunk]

        ttl = extract_turtle_from_texts(
            texts, agent_name=agent_name, model=model, session=session,
            prompt_builder=prompt_builder,
        )

        added = add_turtle_to_store(ttl, store, sources=sources, save=False)
        total_added += added

    return total_added
