"""
RDF Store for Personal Conversational Memory.

Thin wrapper around rdflib.Graph with namespace bindings, persistence,
SPARQL query support, and basic statistics.
"""

from rdflib import Graph, Namespace, Literal, URIRef, RDF, RDFS, XSD
from rdflib.namespace import FOAF, DC, DCTERMS, PROV
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid


# Namespaces used by the PCM ontology
PCM = Namespace("https://w3id.org/pcm/")
SCHEMA = Namespace("http://schema.org/")
OB = Namespace("https://w3id.org/ontobio#")
REL = Namespace("http://purl.org/vocab/relationship/")
BIO = Namespace("http://purl.org/vocab/bio/0.1/")
CV = Namespace("http://rdfs.org/resume-rdf/cv.rdfs#")
OPENCARE = Namespace("https://w3id.org/opencare#")
SIOC = Namespace("http://rdfs.org/sioc/ns#")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

# pcm-claude.ttl namespaces
DOAP = Namespace("http://usefulinc.com/ns/doap#")
NFO = Namespace("http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#")
PPLAN = Namespace("http://purl.org/net/p-plan#")


class RDFMemoryStore:
    """
    RDF-based personal memory store.

    Provides namespace-aware graph persistence, SPARQL querying, and
    statistics.  The extraction pipeline writes Turtle directly into
    ``self.graph``; this class handles serialisation and lookup.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        base_uri: str = "https://w3id.org/pcm/",
        agent_id: Optional[str] = None,
    ):
        self.graph = Graph()
        self.storage_path = storage_path
        self.base_uri = base_uri
        self.agent_id = agent_id or "system"
        self.agent_uri = URIRef(f"{base_uri}agent/{self.agent_id}")

        # Bind namespaces for readable Turtle output
        self.graph.bind("foaf", FOAF)
        self.graph.bind("dc", DC)
        self.graph.bind("dcterms", DCTERMS)
        self.graph.bind("pcm", PCM)
        self.graph.bind("xsd", XSD)
        self.graph.bind("prov", PROV)
        self.graph.bind("schema", SCHEMA)
        self.graph.bind("ob", OB)
        self.graph.bind("rel", REL)
        self.graph.bind("bio", BIO)
        self.graph.bind("cv", CV)
        self.graph.bind("opencare", OPENCARE)
        self.graph.bind("sioc", SIOC)
        self.graph.bind("skos", SKOS)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("doap", DOAP)
        self.graph.bind("nfo", NFO)
        self.graph.bind("p-plan", PPLAN)

        # Load existing graph if storage path exists
        if storage_path:
            try:
                self.graph.parse(storage_path, format="turtle")
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Warning: Could not load graph: {e}")

    def _create_uri(self, identifier: Optional[str] = None) -> URIRef:
        """Create a URI under the base namespace."""
        if identifier:
            return URIRef(f"{self.base_uri}{identifier}")
        return URIRef(f"{self.base_uri}{uuid.uuid4()}")

    def _save(self) -> None:
        """Serialize the graph to disk (Turtle format)."""
        if self.storage_path:
            try:
                self.graph.serialize(destination=self.storage_path, format="turtle")
            except Exception as e:
                print(f"Error saving graph: {e}")

    def export_graph(self, fmt: str = "turtle") -> str:
        """Export the graph as a string in the given format."""
        return self.graph.serialize(format=fmt)

    def sparql_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a SPARQL query and return result bindings as dicts."""
        results = []
        qres = self.graph.query(query)
        for row in qres:
            result = {}
            for var in qres.vars:
                result[str(var)] = str(row[var]) if row[var] else None
            results.append(result)
        return results

    def get_stats(self) -> Dict[str, int]:
        """Return counts of key entity types in the graph."""
        return {
            "total_triples": len(self.graph),
            "people": len(list(self.graph.subjects(RDF.type, FOAF.Person))),
            "actions": len(list(self.graph.subjects(RDF.type, SCHEMA.Action))),
            "events": len(list(self.graph.subjects(RDF.type, SCHEMA.Event))),
            "preferences": len(list(self.graph.subjects(RDF.type, PCM.Preference))),
            "habits": len(list(self.graph.subjects(RDF.type, OB.Habit))),
            "threads": len(list(self.graph.subjects(RDF.type, SIOC.Thread))),
            "posts": len(list(self.graph.subjects(RDF.type, SIOC.Post))),
        }
