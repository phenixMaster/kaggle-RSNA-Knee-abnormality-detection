#!/bin/bash
# Liste des fichiers et dossiers à inclure
FILES="src/ main.py pyproject.toml uv.lock TRAINING_PLAN.md AGENTS.md README.md"

echo "Packaging modified files..."
zip -r dev_bundle.zip $FILES
echo "Bundle created: dev_bundle.zip"
