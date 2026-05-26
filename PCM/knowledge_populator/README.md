# Knowledge Populator

Extract structured knowledge graphs from conversations using LLMs.

For setup instructions, input format, and the KG explorer UI, see the [main README](../README.md).

## Extra options

One TTL file per conversation:

```bash
python -m knowledge_populator -f sample_data/conversations.json --batch-size 50 --per-conversation -o output/
```

Process only the first 3 conversations:

```bash
python -m knowledge_populator -f sample_data/conversations.json --batch-size 50 -o output/kg.ttl --limit 3
```

## Use a different model

Any model supported by [litellm](https://docs.litellm.ai/docs/providers):

```bash
# OpenAI
python -m knowledge_populator -f sample_data/conversations.json --model gpt-4o --batch-size 50 -o output/kg.ttl

# Local (Ollama) — IBM Granite 4 Micro (2.1 GB, fastest)
python -m knowledge_populator -f sample_data/conversations.json --model ollama/ibm/granite4:micro --batch-size 50 --per-conversation --limit 1 -o output/

# Local (Ollama) — IBM Granite 3.3 8B (4.9 GB, better entity recall)
python -m knowledge_populator -f sample_data/conversations.json --model ollama/granite3.3:8b --batch-size 50 --per-conversation --limit 1 -o output/
```

## CLI options

```
python -m knowledge_populator --help

positional arguments:
  text                  Text to extract (or use -f for a file)

options:
  -f, --file FILE       Path to conversations.json file
  -o, --output OUTPUT   Output .ttl file or directory (with --per-conversation) (default: output/kg.ttl)
  -a, --agent AGENT     Name of the person (default: User)
  --model MODEL         LLM model in litellm format (default: anthropic/claude-sonnet-4-5-20250929)
  --batch-size N        User messages per LLM call (default: 1)
  --per-conversation    One TTL file per conversation (default: off, single merged file)
  --limit N             Max conversations to process (default: 0 = all)
  --no-store            Print Turtle to stdout, don't save (default: off)
```

## Python API

```python
from knowledge_populator import (
    RDFMemoryStore,
    load_conversations,
    extract_conversation,
    extract_turtle_from_text,
)

# Single text
ttl = extract_turtle_from_text("I have a dog named Max", agent_name="Alice")
print(ttl)

# Full conversation
store = RDFMemoryStore(storage_path="my_kg.ttl")
conversations = load_conversations("sample_data/conversations.json")
for conv in conversations:
    extract_conversation(conv, store, agent_name="Alice")
store._save()
```
