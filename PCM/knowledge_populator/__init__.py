# Knowledge Populator — extract personal knowledge graphs from conversations

from .extractor import (
    extract_turtle_from_text,
    extract_turtle_from_texts,
    extract_conversation,
    load_conversations,
    add_turtle_to_store,
)
from .rdf_store import RDFMemoryStore
