from langchain_unstructured import UnstructuredLoader

loader = UnstructuredLoader(
    "C:\\Users\\faton\\workspace\\tutorial\\output\\html\\content_section_raw.html",
    mode="elements",  # Element-wise parsing
    unstructured_kwargs={"strategy": "hi_res"}
)

documents = loader.load()

print(f"Loaded {len(documents)} sections from HTML.\n")

for i, doc in enumerate(documents):
    print(f"--- Document #{i+1} ---")
    print("Metadata:", doc.metadata)
    print("Content:\n", doc.page_content[:500])
    print()
