"""Persistent mapping between exact PDF content and document identities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3
import uuid


@dataclass(frozen=True)
class DocumentIdentity:
    fingerprint: str
    document_id: str
    document_dir: Path

    @property
    def structure_path(self) -> Path:
        return self.document_dir / "structure.json"


class DocumentIdentityStore:
    """Resolve PDF bytes to a persistent document identity.

    Older uploads are migrated by hashing their existing ``original.pdf`` files
    and recording their directory names as document IDs.
    """

    def __init__(self, documents_dir: str | Path):
        self.documents_dir = Path(documents_dir)
        self.database_path = self.documents_dir / "document_identity.sqlite3"

    @staticmethod
    def fingerprint(pdf_bytes: bytes) -> str:
        if not isinstance(pdf_bytes, bytes) or not pdf_bytes:
            raise ValueError("PDF content must be non-empty bytes.")
        return hashlib.sha256(pdf_bytes).hexdigest()

    def resolve(self, pdf_bytes: bytes) -> DocumentIdentity:
        fingerprint = self.fingerprint(pdf_bytes)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.database_path, timeout=30) as connection:
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS document_identities (
                    fingerprint TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL UNIQUE
                )
            """)
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT document_id FROM document_identities WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()

            if row:
                document_id = row[0]
            else:
                document_id = self._find_legacy_document(connection, fingerprint)
                if document_id is None:
                    document_id = str(uuid.uuid4())
                connection.execute(
                    "INSERT INTO document_identities (fingerprint, document_id) VALUES (?, ?)",
                    (fingerprint, document_id),
                )

        return DocumentIdentity(
            fingerprint=fingerprint,
            document_id=document_id,
            document_dir=self.documents_dir / document_id,
        )

    def _find_legacy_document(
        self,
        connection: sqlite3.Connection,
        fingerprint: str,
    ) -> str | None:
        for pdf_path in self.documents_dir.glob("*/original.pdf"):
            if pdf_path.parent.name == self.database_path.stem:
                continue
            existing_fingerprint = _fingerprint_file(pdf_path)
            if existing_fingerprint != fingerprint:
                continue
            document_id = pdf_path.parent.name
            existing = connection.execute(
                "SELECT fingerprint FROM document_identities WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            if existing is None or existing[0] == fingerprint:
                return document_id
        return None


def _fingerprint_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
