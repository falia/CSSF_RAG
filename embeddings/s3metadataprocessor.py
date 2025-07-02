import json
import boto3

from typing import List, Dict, Any
from langchain_core.documents import Document

import hashlib
import logging

from embedding_provider.embedding_provider import EmbeddingService
from chunker.document_chunker import DocumentChunker
from parsers.parser import EurlexHTMLParser, CSSFHTMLParser, PDFParser, DocumentProcessor


class S3MetadataProcessor:

    def __init__(self, s3_bucket: str, session_id: str = None, milvus_config: Dict = None):
        self.s3_bucket = s3_bucket
        self.session_id = session_id
        self.s3 = boto3.client("s3")

        # Initialize processors similar to your UrlSpider
        self.processor = DocumentProcessor(parsers=[EurlexHTMLParser(), CSSFHTMLParser(), PDFParser()])
        self.chunker = DocumentChunker(max_chunk_size=1800, overlap=200)
        self.seen_hashes = set()

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize Milvus config
        if milvus_config is None:
            milvus_config = {
                'host': '34.241.177.15',
                'port': '19530',
                'collection_name': 'cssf_documents_final',
                'connection_args': {"host": "34.241.177.15", "port": "19530"}
            }

        # Initialize embedding service
        self.embedding_service = EmbeddingService(
            use_remote=True,
            use_tei=True,
            milvus_config=milvus_config,
            endpoint_name='embedding-endpoint',
            region_name='eu-west-1'
        )

    def list_sessions(self) -> List[str]:
        """List all available crawl sessions in S3, sorted by timestamp."""
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix="",
                Delimiter="/"
            )

            sessions = []
            for prefix in response.get('CommonPrefixes', []):
                session_id = prefix['Prefix'].rstrip('/')
                sessions.append(session_id)

            # Sort sessions by timestamp (assuming format: YYYYMMDD_HHMMSS)
            sessions.sort(reverse=True)  # Most recent first
            return sessions
        except Exception as e:
            self.logger.error(f"Error listing sessions: {e}")
            return []

    def get_most_recent_session(self) -> str:
        """Get the most recent crawl session ID."""
        sessions = self.list_sessions()
        if not sessions:
            self.logger.warning("No sessions found")
            return None

        most_recent = sessions[0]  # Already sorted with most recent first
        self.logger.info(f"Most recent session: {most_recent}")
        return most_recent

    def get_session_metadata_files(self, session_id: str) -> List[str]:
        """Get all metadata.json files for a specific session using pagination."""
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.s3_bucket,
                Prefix=f"{session_id}/"
            )

            metadata_files = []
            for page in page_iterator:
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith('metadata.json'):
                        metadata_files.append(obj['Key'])

            return metadata_files

        except Exception as e:
            self.logger.error(f"Error listing metadata files for session {session_id}: {e}", exc_info=True)
            return []

    def read_metadata_from_s3(self, s3_key: str) -> Dict[str, Any]:
        """Read metadata from S3."""
        try:
            response = self.s3.get_object(Bucket=self.s3_bucket, Key=s3_key)
            metadata = json.loads(response['Body'].read().decode('utf-8'))
            return metadata
        except Exception as e:
            self.logger.error(f"Error reading metadata from {s3_key}: {e}")
            return {}

    def download_document_from_s3(self, s3_uri: str) -> bytes:
        """Download document content from S3."""
        try:
            # Parse S3 URI: s3://bucket/key
            s3_key = s3_uri.replace(f"s3://{self.s3_bucket}/", "")
            response = self.s3.get_object(Bucket=self.s3_bucket, Key=s3_key)
            return response['Body'].read()
        except Exception as e:
            self.logger.error(f"Error downloading document from {s3_uri}: {e}")
            return b""

    def hash_document(self, doc: Document) -> str:
        """Generate hash for document deduplication."""
        base = doc.page_content + str(doc.metadata.get("source", ""))
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def flatten_metadata_for_search(self, metadata):
        """
        Flatten metadata to make important fields searchable while preserving complex data
        """
        flattened = {
            # Direct searchable fields (frequently queried)
            'doc_id': metadata.get('doc_id', ''),
            'title': metadata.get('title', ''),
            'document_type': metadata.get('document_type', ''),
            'document_number': metadata.get('document_number', ''),
            'lang': metadata.get('lang', ''),
            'super_category': metadata.get('super_category', ''),
            'subtitle': metadata.get('subtitle', ''),
            'url': metadata.get('url', ''),
            'crawl_timestamp': metadata.get('crawl_timestamp', ''),
            'file_size': metadata.get('file_size', 0),
            'publication_date': metadata.get('publication_date') or '',
            'update_date': metadata.get('update_date') or '',
            'content_hash': metadata.get('content_hash', ''),

            # Flatten important arrays to searchable pipe-separated strings
            'entities_text': ' | '.join(metadata.get('entities', [])),
            'keywords_text': ' | '.join(metadata.get('keywords', [])),
            'themes_text': ' | '.join(metadata.get('themes', [])),

            # Bundle complex/nested data into JSON for when you need full details
            'complex_metadata': json.dumps({
                'content_hash': metadata.get('content_hash', ''),
                'top_related': metadata.get('top_related', []),
                'bottom_related': metadata.get('bottom_related', []),
                'entities': metadata.get('entities', []),  # Keep original arrays too
                'keywords': metadata.get('keywords', []),
                'themes': metadata.get('themes', []),
                # Add any other complex fields here
                'crawl_session': metadata.get('crawl_session', ''),
                'source': metadata.get('source', ''),
                'processing_metadata': metadata.get('processing_metadata', {})
            })
        }

        return flattened

    def sanitize_metadata_json(self, metadata):
        """Legacy method - now replaced by flatten_metadata_for_search"""
        return self.flatten_metadata_for_search(metadata)

    import json

    def convert_elements_to_documents(self, elements, metadata: Dict[str, Any]) -> List[Document]:
        """Convert unstructured elements to LangChain Documents."""
        documents = []

        for i, element in enumerate(elements):
            try:
                # Extract text content
                if hasattr(element, 'text') and element.text:
                    text_content = element.text.strip()
                elif hasattr(element, 'page_content'):
                    text_content = element.page_content.strip()
                else:
                    text_content = str(element).strip()

                if not text_content:
                    continue

                # Safely handle metadata - ensure it's a dictionary
                if isinstance(metadata, dict):
                    doc_metadata = metadata.copy()
                else:
                    self.logger.warning(f"Expected dict for metadata, got {type(metadata)}. Using empty dict.")
                    doc_metadata = {}

                # Handle complex_metadata - it might be a JSON string
                complex_metadata_dict = {}
                if 'complex_metadata' in doc_metadata:
                    if isinstance(doc_metadata['complex_metadata'], str):
                        # Parse JSON string to dict
                        try:
                            complex_metadata_dict = json.loads(doc_metadata['complex_metadata'])
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Failed to parse complex_metadata JSON: {e}")
                            complex_metadata_dict = {}
                    elif isinstance(doc_metadata['complex_metadata'], dict):
                        complex_metadata_dict = doc_metadata['complex_metadata'].copy()

                # Only extract page_number from unstructured element metadata
                if hasattr(element, 'metadata') and element.metadata:
                    try:
                        if isinstance(element.metadata, dict):
                            # Only keep page_number
                            if 'page_number' in element.metadata:
                                doc_metadata['page_number'] = element.metadata['page_number']

                            # Store all other element metadata in complex_metadata
                            other_metadata = {k: v for k, v in element.metadata.items() if k != 'page_number'}
                            if other_metadata:
                                complex_metadata_dict['element_metadata'] = other_metadata
                    except Exception as e:
                        self.logger.warning(f"Error processing element metadata for element {i}: {e}")

                # Add element index to complex metadata
                complex_metadata_dict['element_index'] = i

                # Convert complex_metadata back to JSON string for Milvus
                doc_metadata['complex_metadata'] = json.dumps(complex_metadata_dict)

                # Create LangChain Document
                doc = Document(
                    page_content=text_content,
                    metadata=doc_metadata
                )

                documents.append(doc)

            except Exception as e:
                self.logger.error(f"Error converting element {i} to document: {e}")
                # Create a minimal document to avoid losing content
                try:
                    minimal_doc = Document(
                        page_content=text_content if 'text_content' in locals() else str(element),
                        metadata={
                            'element_index': i,
                            'error': str(e),
                            'complex_metadata': json.dumps({'element_index': i, 'error': str(e)})
                        }
                    )
                    documents.append(minimal_doc)
                except:
                    self.logger.error(f"Failed to create minimal document for element {i}")
                continue

        return documents

    def process_document(self, metadata: Dict[str, Any]) -> List[Document]:
        """Process a single document with its metadata and referenced files."""
        all_documents = []

        # Process each referenced document
        for file_info in metadata.get('top_related', []):
            try:
                s3_uri = file_info.get('s3_uri')
                original_url = file_info.get('url')
                content_type = file_info.get('content_type', 'application/octet-stream')

                if not s3_uri:
                    continue

                self.logger.info(f"Processing document: {original_url}")

                # Download document from S3
                content = self.download_document_from_s3(s3_uri)
                if not content:
                    continue

                # Process document using your existing processor with content and params
                elements = self.processor.process(content, original_url, content_type)

                # Flatten metadata for Milvus compatibility and searchability
                flattened_metadata = self.flatten_metadata_for_search(metadata)

                # Option 1: Convert elements to Documents (if chunking is disabled)
                documents = self.convert_elements_to_documents(elements, flattened_metadata)
                all_documents.extend(documents)

                # Option 2: Use chunking (uncomment if you want to enable chunking)
                # chunked_docs = self.chunker.chunk_document(elements, original_url)
                # for doc in chunked_docs:
                #     # Add the flattened metadata to each chunk
                #     doc.metadata.update(flattened_metadata)
                # all_documents.extend(chunked_docs)

            except Exception as e:
                self.logger.error(f"Error processing document {file_info.get('url', 'unknown')}: {e}")

        return all_documents

    def store_documents_in_milvus(self, documents: List[Document]) -> Dict[str, Any]:
        """Store documents in Milvus with deduplication."""
        if not documents:
            return {'count': 0, 'milvus_ids': []}

        # Deduplication
        new_docs = []
        texts_to_store = []
        metadatas_to_store = []

        for doc in documents:
            doc_id = self.hash_document(doc)
            if doc_id in self.seen_hashes:
                continue

            # Add doc_id to metadata (this will overwrite the original doc_id if it exists)
            doc.metadata["doc_id"] = doc_id
            self.seen_hashes.add(doc_id)
            new_docs.append(doc)

            # Prepare for batch storage - now safe because doc is a Document object
            texts_to_store.append(doc.page_content)
            metadatas_to_store.append(doc.metadata)

        # Store in Milvus
        if new_docs:
            try:
                result = self.embedding_service.add_texts_to_store(
                    texts=texts_to_store,
                    metadatas=metadatas_to_store
                )
                self.logger.info(f"Stored {result['count']} new documents in Milvus")
                return result
            except Exception as e:
                self.logger.error(f"Failed to store documents in Milvus: {e}")
                return {'count': 0, 'milvus_ids': []}

        return {'count': 0, 'milvus_ids': []}

    def process_session(self, session_id: str) -> Dict[str, Any]:
        self.logger.info(f"Processing session: {session_id}")

        # Get all metadata files for this session
        metadata_files = self.get_session_metadata_files(session_id)

        print("Meta data files to process ---------------------------------------------------------", len(metadata_files))

        if not metadata_files:
            self.logger.warning(f"No metadata files found for session {session_id}")
            return {'processed': 0, 'stored': 0, 'errors': 0}

        total_processed = 0
        total_stored = 0
        total_errors = 0

        for metadata_file in metadata_files:
            try:
                # Read metadata
                metadata = self.read_metadata_from_s3(metadata_file)
                if not metadata:
                    continue

                # Add session info to metadata
                metadata['crawl_session'] = session_id

                # Process documents
                documents = self.process_document(metadata)
                total_processed += len(documents)

                # Store in Milvus
                result = self.store_documents_in_milvus(documents)
                total_stored += result['count']

                self.logger.info(f"Processed {metadata_file}: {len(documents)} docs, {result['count']} stored")

            except Exception as e:
                self.logger.error(f"Error processing {metadata_file}: {e}")
                total_errors += 1

        summary = {
            'session_id': session_id,
            'processed': total_processed,
            'stored': total_stored,
            'errors': total_errors
        }

        self.logger.info(f"Session {session_id} complete: {summary}")
        return summary

    def process_latest_session(self) -> Dict[str, Any]:
        """Process the most recent crawl session."""
        latest_session = self.get_most_recent_session()
        if not latest_session:
            self.logger.warning("No sessions found")
            return {'processed': 0, 'stored': 0, 'errors': 0}

        return self.process_session(latest_session)


def main():
    S3_BUCKET = "cssf-crawl"
    SESSION_ID = None
    MILVUS_CONFIG = {
        'host': '34.241.177.15',
        'port': '19530',
        'collection_name': 'cssf_documents_final_demo_3',
        'connection_args': {"host": "34.241.177.15", "port": "19530"}
    }

    processor = S3MetadataProcessor(
        s3_bucket=S3_BUCKET,
        session_id=SESSION_ID,
        milvus_config=MILVUS_CONFIG
    )

    sessions = processor.list_sessions()
    print(f"Available sessions: {sessions}")

    most_recent = processor.get_most_recent_session()
    if most_recent:
        print(f"Most recent session: {most_recent}")

        # Process the most recent session
        result = processor.process_session(most_recent)
        print(f"Processing complete: {result}")
    else:
        print("No sessions found to process")

    # Alternative processing options:
    # 1. Process specific session:
    # result = processor.process_session("20250629_143022")


if __name__ == "__main__":
    main()