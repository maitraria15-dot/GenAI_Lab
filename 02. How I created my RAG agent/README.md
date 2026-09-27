# 1. Create the new enterprise folder structure
mkdir -p src/enterprise_rag/core src/enterprise_rag/db src/enterprise_rag/tools

# 2. Add empty __init__.py files for package discovery
touch src/enterprise_rag/__init__.py<br>
touch src/enterprise_rag/core/__init__.py<br>
touch src/enterprise_rag/db/__init__.py<br>
touch src/enterprise_rag/tools/__init__.py<br>


# 📦 Understanding `__init__.py` in Python

## 1. What is `__init__.py`?

In Python, `__init__.py` is a special file used to mark directories as **Python packages**. 

When Python encounters an `__init__.py` file inside a folder, it recognizes that folder as an importable package namespace. This allows you to write clean relative and absolute imports across your project modules.

---

## 2. How `__init__.py` Is Created in Production Layouts

In enterprise projects following the `src/` directory layout, subpackages are created and initialized using terminal setup commands:

```bash
# Create directory structure
mkdir -p src/enterprise_rag/core src/enterprise_rag/db src/enterprise_rag/tools tests

# Create empty __init__.py files in each package folder
touch src/enterprise_rag/__init__.py
touch src/enterprise_rag/core/__init__.py
touch src/enterprise_rag/db/__init__.py
touch src/enterprise_rag/tools/__init__.py
touch tests/__init__.py

** ## 3. Why **
