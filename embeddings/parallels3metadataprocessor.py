import json
import boto3
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any
from langchain_core.documents import Document
import hashlib
import logging
import time
import os
from threading import Lock

from embedding_provider.embedding_provider import EmbeddingService
from chunker.document_chunker import DocumentChunker
from parsers.parser import EurlexHTMLParser, CSSFHTMLParser, PDFParser, DocumentProcessor


class ParallelS3MetadataProcessor:
    """Simple parallelized version of your existing S3MetadataProcessor"""

    def __init__(self, s3_bucket: str, session_id: str = None, milvus_config: Dict = None,
                 max_workers: int = None):
        self.s3_bucket = s3_bucket
        self.session_id = session_id
        self.s3 = boto3.client("s3")

        # Determine optimal worker count
        cpu_count = multiprocessing.cpu_count()

        # Since we have at most 2 documents per metadata file, use same workers for both levels
        if max_workers is None:
            max_workers = min(cpu_count, 32)  # Good balance for both metadata and document processing
        self.max_workers = max_workers

        # Thread-safe components
        self.seen_hashes = set()
        self.hash_lock = Lock()

        # Initialize processors (will create per-worker instances)
        self.processor = DocumentProcessor(parsers=[EurlexHTMLParser(), CSSFHTMLParser(), PDFParser()])
        self.chunker = DocumentChunker(max_chunk_size=1800, overlap=200)

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize Milvus config
        if milvus_config is None:
            milvus_config = {
                'host': '34.241.177.15',
                'port': '19530',
                'collection_name': 'cssf_documents_final_test1234',
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

        print(f"Initialized with {self.max_workers} workers for both metadata and document processing")
        print(f"CPU cores available: {cpu_count}")
        print(f"Optimized for ~2 documents per metadata file")

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

        most_recent = sessions[0]
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
        """Flatten metadata to make important fields searchable while preserving complex data"""
        flattened = {
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
            'entities_text': ' | '.join(metadata.get('entities', [])),
            'keywords_text': ' | '.join(metadata.get('keywords', [])),
            'themes_text': ' | '.join(metadata.get('themes', [])),
            'complex_metadata': json.dumps({
                'content_hash': metadata.get('content_hash', ''),
                'top_related': metadata.get('top_related', []),
                'bottom_related': metadata.get('bottom_related', []),
                'entities': metadata.get('entities', []),
                'keywords': metadata.get('keywords', []),
                'themes': metadata.get('themes', []),
                'crawl_session': metadata.get('crawl_session', ''),
                'source': metadata.get('source', ''),
                'processing_metadata': metadata.get('processing_metadata', {})
            })
        }
        return flattened

    def convert_elements_to_documents(self, elements, metadata: Dict[str, Any]) -> List[Document]:
        """Convert unstructured elements to LangChain Documents."""
        documents = []

        for i, element in enumerate(elements):
            try:
                if hasattr(element, 'text') and element.text:
                    text_content = element.text.strip()
                elif hasattr(element, 'page_content'):
                    text_content = element.page_content.strip()
                else:
                    text_content = str(element).strip()

                if not text_content:
                    continue

                if isinstance(metadata, dict):
                    doc_metadata = metadata.copy()
                else:
                    doc_metadata = {}

                complex_metadata_dict = {}
                if 'complex_metadata' in doc_metadata:
                    if isinstance(doc_metadata['complex_metadata'], str):
                        try:
                            complex_metadata_dict = json.loads(doc_metadata['complex_metadata'])
                        except json.JSONDecodeError:
                            complex_metadata_dict = {}
                    elif isinstance(doc_metadata['complex_metadata'], dict):
                        complex_metadata_dict = doc_metadata['complex_metadata'].copy()

                if hasattr(element, 'metadata') and element.metadata:
                    try:
                        if isinstance(element.metadata, dict):
                            if 'page_number' in element.metadata:
                                doc_metadata['page_number'] = element.metadata['page_number']
                            other_metadata = {k: v for k, v in element.metadata.items() if k != 'page_number'}
                            if other_metadata:
                                complex_metadata_dict['element_metadata'] = other_metadata
                    except Exception as e:
                        self.logger.warning(f"Error processing element metadata for element {i}: {e}")

                complex_metadata_dict['element_index'] = i
                doc_metadata['complex_metadata'] = json.dumps(complex_metadata_dict)

                doc = Document(page_content=text_content, metadata=doc_metadata)
                documents.append(doc)

            except Exception as e:
                self.logger.error(f"Error converting element {i} to document: {e}")
                continue

        return documents

    def process_single_file(self, file_info: Dict, metadata: Dict[str, Any]) -> List[Document]:
        """Process a single file - this function will be called in parallel"""
        try:
            s3_uri = file_info.get('s3_uri')
            original_url = file_info.get('url')
            content_type = file_info.get('content_type', 'application/octet-stream')

            if not s3_uri:
                return []

            # Download document from S3
            content = self.download_document_from_s3(s3_uri)
            if not content:
                return []

            # Process document using your existing processor
            elements = self.processor.process(content, original_url, content_type)

            # Flatten metadata for Milvus compatibility
            flattened_metadata = self.flatten_metadata_for_search(metadata)

            # Convert elements to Documents
            documents = self.convert_elements_to_documents(elements, flattened_metadata)

            self.logger.info(f"Processed {original_url}: {len(documents)} documents")
            return documents

        except Exception as e:
            self.logger.error(f"Error processing document {file_info.get('url', 'unknown')}: {e}")
            return []

    def process_document_parallel(self, metadata: Dict[str, Any]) -> List[Document]:
        """Process all documents in metadata using parallel processing - NEW PARALLEL VERSION"""
        all_documents = []

        # Get all files to process
        files_to_process = metadata.get('top_related', [])

        if not files_to_process:
            return all_documents

        print(f"Processing {len(files_to_process)} files in parallel (max 2 expected)...")

        # Process files in parallel using ThreadPoolExecutor
        # Since we have at most 2 files, use same max_workers as metadata level
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(files_to_process))) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(self.process_single_file, file_info, metadata): file_info
                for file_info in files_to_process
            }

            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_file), 1):
                file_info = future_to_file[future]
                try:
                    documents = future.result()
                    all_documents.extend(documents)
                    print(
                        f"  Completed {i}/{len(files_to_process)}: {file_info.get('url', 'unknown')} -> {len(documents)} docs")
                except Exception as e:
                    self.logger.error(f"Error processing {file_info.get('url', 'unknown')}: {e}")

        return all_documents

    def store_documents_in_milvus(self, documents: List[Document]) -> Dict[str, Any]:
        """Store documents in Milvus with deduplication - made thread-safe"""
        if not documents:
            return {'count': 0, 'milvus_ids': []}

        # Thread-safe deduplication
        new_docs = []
        texts_to_store = []
        metadatas_to_store = []

        with self.hash_lock:  # Thread-safe access to seen_hashes
            for doc in documents:
                doc_id = self.hash_document(doc)
                if doc_id in self.seen_hashes:
                    continue

                doc.metadata["doc_id"] = doc_id
                self.seen_hashes.add(doc_id)
                new_docs.append(doc)
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

    def process_metadata_file(self, metadata_file: str, session_id: str) -> Dict[str, Any]:
        """Process a single metadata file - this function will be called in parallel"""
        try:
            print(f"Processing metadata file: {metadata_file}")

            # Read metadata
            metadata = self.read_metadata_from_s3(metadata_file)
            if not metadata:
                return {'processed': 0, 'stored': 0, 'errors': 1, 'file': metadata_file}

            # Add session info to metadata
            metadata['crawl_session'] = session_id

            # Process documents in parallel (this calls the parallel version)
            documents = self.process_document_parallel(metadata)

            # Store in Milvus
            result = self.store_documents_in_milvus(documents)

            print(f"Completed {metadata_file}: {len(documents)} docs processed, {result['count']} stored")

            return {
                'processed': len(documents),
                'stored': result['count'],
                'errors': 0,
                'file': metadata_file
            }

        except Exception as e:
            self.logger.error(f"Error processing {metadata_file}: {e}")
            return {'processed': 0, 'stored': 0, 'errors': 1, 'file': metadata_file}

    def process_session_parallel(self, session_id: str) -> Dict[str, Any]:
        """Process session with parallel metadata file processing - NEW PARALLEL VERSION"""
        self.logger.info(f"Processing session: {session_id}")
        start_time = time.time()

        # Get all metadata files for this session
        metadata_files = self.get_session_metadata_files(session_id)

        print(f"Found {len(metadata_files)} metadata files to process")

        if not metadata_files:
            self.logger.warning(f"No metadata files found for session {session_id}")
            return {'processed': 0, 'stored': 0, 'errors': 0}

        total_processed = 0
        total_stored = 0
        total_errors = 0

        # Process metadata files in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all metadata files for processing
            future_to_file = {
                executor.submit(self.process_metadata_file, metadata_file, session_id): metadata_file
                for metadata_file in metadata_files
            }

            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_file), 1):
                metadata_file = future_to_file[future]
                try:
                    result = future.result()
                    total_processed += result['processed']
                    total_stored += result['stored']
                    total_errors += result['errors']

                    elapsed = time.time() - start_time
                    print(f"Progress: {i}/{len(metadata_files)} ({i / len(metadata_files) * 100:.1f}%) - "
                          f"Elapsed: {elapsed:.1f}s - "
                          f"Total processed: {total_processed}, stored: {total_stored}")

                except Exception as e:
                    self.logger.error(f"Error processing {metadata_file}: {e}")
                    total_errors += 1

        elapsed_time = time.time() - start_time
        summary = {
            'session_id': session_id,
            'processed': total_processed,
            'stored': total_stored,
            'errors': total_errors,
            'elapsed_time': elapsed_time,
            'docs_per_second': total_processed / elapsed_time if elapsed_time > 0 else 0
        }

        print(f"\n=== SESSION COMPLETE ===")
        print(f"Session: {session_id}")
        print(f"Metadata files: {len(metadata_files)}")
        print(f"Documents processed: {total_processed}")
        print(f"Documents stored: {total_stored}")
        print(f"Errors: {total_errors}")
        print(f"Time: {elapsed_time:.1f} seconds")
        print(f"Speed: {summary['docs_per_second']:.1f} docs/second")

        return summary

    def process_latest_session_parallel(self) -> Dict[str, Any]:
        """Process the most recent crawl session with parallelization - NEW PARALLEL VERSION"""
        latest_session = self.get_most_recent_session()
        if not latest_session:
            self.logger.warning("No sessions found")
            return {'processed': 0, 'stored': 0, 'errors': 0}

        return self.process_session_parallel(latest_session)


def main():
    """Main function - simple changes to use parallel processing"""
    S3_BUCKET = "cssf-crawl"
    SESSION_ID = None
    MILVUS_CONFIG = {
        'host': '34.241.177.15',
        'port': '19530',
        'collection_name': 'cssf_documents_final_demo_3',
        'connection_args': {"host": "34.241.177.15", "port": "19530"}
    }

    # Create parallel processor with single worker pool
    processor = ParallelS3MetadataProcessor(
        s3_bucket=S3_BUCKET,
        session_id=SESSION_ID,
        milvus_config=MILVUS_CONFIG,
        max_workers=6  # Adjust based on your machine (good for both metadata and docs)
    )

    sessions = processor.list_sessions()
    print(f"Available sessions: {sessions}")

    most_recent = processor.get_most_recent_session()
    if most_recent:
        print(f"Most recent session: {most_recent}")

        # Use the parallel processing method
        result = processor.process_session_parallel(most_recent)
        print(f"Processing complete: {result}")
    else:
        print("No sessions found to process")


if __name__ == "__main__":
    main()