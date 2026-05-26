"""
CLI entry point for the PCM explorer.

Usage:
    python -m ui --kg-dir output/ --conversations sample_data/conversations.json
    python -m ui --kg-dir output/ --port 8080
"""

from .explorer import main

if __name__ == "__main__":
    main()
