# KG Explorer

Web UI for browsing, querying, and visualizing extracted knowledge graphs.

For setup instructions and an overview, see the [main README](../README.md).

## CLI options

```
python -m ui --help

options:
  --kg-dir DIR            Directory with per-conversation TTL files (required)
  --conversations FILE    Path to conversations.json for message viewing (optional)
  --port PORT             Server port (default: 5050)
```

> **Note:** The explorer expects `--per-conversation` output (one TTL per conversation). If you extracted into a single `kg.ttl`, re-run with `--per-conversation -o output/`.
