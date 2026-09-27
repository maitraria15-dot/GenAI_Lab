# 1. Create the new enterprise folder structure
mkdir -p src/enterprise_rag/core src/enterprise_rag/db src/enterprise_rag/tools

# 2. Add empty __init__.py files for package discovery
touch src/enterprise_rag/__init__.py/n
* touch src/enterprise_rag/core/__init__.py
touch src/enterprise_rag/db/__init__.py
touch src/enterprise_rag/tools/__init__.py

# 3. Clean up root directory files (delete or move rag_assistant.py into src)
rm rag_assistant.py rag_assistant.ipynb
