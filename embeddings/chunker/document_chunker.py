
from langchain_core.documents import Document

from unstructured.chunking.title import chunk_by_title
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentChunker:
    def __init__(self, max_chunk_size=1800, overlap=200):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]  # Prioritize paragraph breaks
        )

    def chunk_document(self, elements, source_url):
        # Step 1: Use title-based chunking to respect document structure
        title_chunks = chunk_by_title(
            elements,
            max_characters=self.max_chunk_size,  # Respect size limits
            new_after_n_chars=int(self.max_chunk_size * 0.8),  # Start looking for breaks at 80%
            combine_text_under_n_chars=200,  # Combine very small sections
        )

        processed_chunks = []

        for chunk in title_chunks:
            if not isinstance(chunk.text, str) or not chunk.text.strip():
                continue

            chunk_text = chunk.text.strip()

            # Step 2: If title chunk is still too large, split it further
            if len(chunk_text) > self.max_chunk_size:
                # Use recursive splitter as fallback for oversized title chunks
                sub_chunks = self.fallback_splitter.create_documents([chunk_text])

                for i, sub_chunk in enumerate(sub_chunks):
                    metadata = {
                        "source_url": source_url,
                        "chunk_type": "title_subsection",
                        "subsection_index": i,
                        "is_split_chunk": True
                    }

                    # Preserve any original metadata from the title chunk
                    if hasattr(chunk, 'metadata') and chunk.metadata:
                        if hasattr(chunk.metadata, '__dict__'):
                            metadata.update(chunk.metadata.__dict__)
                        elif isinstance(chunk.metadata, dict):
                            metadata.update(chunk.metadata)

                    processed_chunks.append(Document(
                        page_content=sub_chunk.page_content,
                        metadata=metadata
                    ))
            else:
                # Step 3: Keep title-based chunks that are appropriately sized
                metadata = {
                    "source_url": source_url,
                    "chunk_type": "title_section",
                    "is_split_chunk": False
                }

                # Extract title information if available
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    if hasattr(chunk.metadata, '__dict__'):
                        metadata.update(chunk.metadata.__dict__)
                    elif isinstance(chunk.metadata, dict):
                        metadata.update(chunk.metadata)

                processed_chunks.append(Document(
                    page_content=chunk_text,
                    metadata=metadata
                ))

        return processed_chunks