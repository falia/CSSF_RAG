# debug_import.py (save in project root)
import sys
import os

print("=== DEBUGGING IMPORT ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries

# Check directory structure
print(f"\nProject root contents: {os.listdir('.')}")
print(f"embedding_provider contents: {os.listdir('embeddings/embedding_provider')}")

# Check __init__.py file
init_file = "embeddings/embedding_provider/__init__.py"
print(f"\n__init__.py exists: {os.path.exists(init_file)}")
print(f"__init__.py size: {os.path.getsize(init_file)} bytes")

# Try importing step by step
print("\n=== IMPORT TEST ===")
try:
    from embeddings import embedding_provider

    print("✅ Can import embedding_provider package")
    print(f"Package location: {embedding_provider.__file__}")
except Exception as e:
    print(f"❌ Cannot import package: {e}")
    sys.exit(1)

try:
    from embeddings.embedding_provider import embedding_provider as ep_module

    print("✅ Can import embedding_provider module")
except Exception as e:
    print(f"❌ Cannot import module: {e}")
    sys.exit(1)

try:
    from embeddings.embedding_provider import EmbeddingService
    print("✅ Can import EmbeddingService class")
    print("SUCCESS! Import works!")
except Exception as e:
    print(f"❌ Cannot import class: {e}")