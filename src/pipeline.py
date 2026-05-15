"""
Main Pipeline
-------------
Orchestrates the full workflow:
Document Processing → Retrieval → Draft Generation → Edit Learning
"""

from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .document_processor import DocumentProcessor
from .retrieval import RetrievalLayer
from .draft_generator import DraftGenerator
from .edit_learner import EditLearner

console = Console()


class LegalDraftPipeline:
    """
    End-to-end pipeline for legal document drafting.
    
    Usage:
        pipeline = LegalDraftPipeline()
        pipeline.ingest("contract.pdf")
        result = pipeline.draft("summary", "Summarize the key obligations")
        pipeline.record_edit(result['draft'], edited_version, "summary")
    """

    def __init__(self, db_dir: str = "./chroma_db", memory_path: str = "./edit_memory.json"):
        self.processor = DocumentProcessor()
        self.retrieval = RetrievalLayer(persist_dir=db_dir)
        self.generator = DraftGenerator()
        self.learner = EditLearner(memory_path=memory_path)
        self._ingested_docs = []

    def ingest(self, file_path: str) -> dict:
        """Process and index a document."""
        console.print(f"[bold blue]📄 Processing:[/bold blue] {file_path}")
        
        # Step 1: Extract text
        doc = self.processor.process(file_path)
        
        if doc.warnings:
            for w in doc.warnings:
                console.print(f"  [yellow]⚠ {w}[/yellow]")
        
        console.print(f"  ✓ Extraction method: [cyan]{doc.extraction_method}[/cyan]")
        console.print(f"  ✓ Confidence: [cyan]{doc.confidence:.0%}[/cyan]")
        console.print(f"  ✓ Text length: [cyan]{len(doc.cleaned_text):,} chars[/cyan]")
        
        if doc.structured_fields:
            console.print(f"  ✓ Structured fields: {list(doc.structured_fields.keys())}")
        
        # Step 2: Chunk and embed
        chunks_added = self.retrieval.ingest_document(doc)
        console.print(f"  ✓ Indexed [cyan]{chunks_added}[/cyan] chunks\n")
        
        self._ingested_docs.append(doc)
        
        return {
            "source": file_path,
            "extraction_method": doc.extraction_method,
            "confidence": doc.confidence,
            "chunks_indexed": chunks_added,
            "structured_fields": doc.structured_fields,
            "warnings": doc.warnings
        }

    def draft(
        self,
        draft_type: str,
        task_description: str,
        top_k: int = 5,
        extra_instructions: str = ""
    ) -> dict:
        """Generate a grounded draft."""
        
        console.print(f"[bold blue]✍ Generating {draft_type}...[/bold blue]")
        
        # Step 3: Retrieve relevant evidence
        chunks = self.retrieval.retrieve(task_description, top_k=top_k)
        console.print(f"  ✓ Retrieved [cyan]{len(chunks)}[/cyan] evidence chunks")
        
        # Get structured fields from all ingested docs
        all_fields = {}
        for doc in self._ingested_docs:
            for k, v in doc.structured_fields.items():
                all_fields.setdefault(k, []).extend(v)
        
        # Get learned style guidelines
        style_guidelines = self.learner.get_style_guidelines(draft_type)
        if style_guidelines:
            console.print(f"  ✓ Applying [cyan]{len(style_guidelines.splitlines())-1}[/cyan] learned style preferences")
        
        # Step 4: Generate draft
        result = self.generator.generate(
            draft_type=draft_type,
            retrieved_chunks=chunks,
            structured_fields=all_fields,
            task_description=task_description,
            style_guidelines=style_guidelines,
            extra_instructions=extra_instructions
        )
        
        grounding = result["grounding_report"]
        console.print(
            f"  ✓ Grounding score: [cyan]{grounding['score']:.0%}[/cyan] "
            f"— {grounding.get('assessment', '')}"
        )
        console.print(f"  ✓ Draft generated ({result['usage']['output_tokens']} tokens)\n")
        
        return result

    def record_edit(self, original: str, edited: str, draft_type: str) -> dict:
        """Record an operator edit and update learning."""
        console.print("[bold blue]📝 Recording operator edit...[/bold blue]")
        
        record = self.learner.record_edit(original, edited, draft_type)
        
        console.print(f"  ✓ {len(record.changes)} changes detected")
        console.print(f"  ✓ {len(record.extracted_patterns)} patterns learned")
        
        if record.extracted_patterns:
            for pattern in record.extracted_patterns:
                console.print(f"    → {pattern}")
        
        console.print()
        return {
            "changes_detected": len(record.changes),
            "patterns_learned": record.extracted_patterns,
            "total_edit_memory": self.learner.get_stats()
        }

    def show_stats(self):
        """Display pipeline statistics."""
        stats = self.learner.get_stats()
        
        table = Table(title="Edit Learning Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Edits Recorded", str(stats["total_edits_recorded"]))
        table.add_row("Draft Types Learned", ", ".join(stats["draft_types_learned"]) or "None")
        table.add_row("Last Updated", stats["last_updated"] or "Never")
        
        console.print(table)