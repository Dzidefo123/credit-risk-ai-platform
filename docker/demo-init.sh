#!/bin/sh
# One-off demo initialisation (SYNTHETIC): pipeline -> model -> promotion -> monitoring -> demo API keys.
set -e
DS=synthetic_fixture
# Fictional underwriting document packets for the Documents page (deterministic; cheap to regenerate).
crisk docai-generate --dataset "$DS" --out data/underwriting_docs_v1 > /dev/null
if [ -f "artifacts/$DS/models/champion.json" ] && [ -f "artifacts/$DS/DEMO_KEYS.txt" ]; then
  echo "Demo already initialised."; cat "artifacts/$DS/DEMO_KEYS.txt"; exit 0
fi
crisk run-slice --dataset "$DS" --n-random 20000 > /dev/null
crisk monitor --dataset "$DS" --eval-dataset "$DS" > /dev/null
ANALYST=$(crisk create-api-key --dataset "$DS" --name demo-analyst --role analyst | python -c "import json,sys;print(json.load(sys.stdin)['api_key'])")
AUDITOR=$(crisk create-api-key --dataset "$DS" --name demo-auditor --role auditor | python -c "import json,sys;print(json.load(sys.stdin)['api_key'])")
cat > "artifacts/$DS/DEMO_KEYS.txt" <<KEYS
=== DEMO API KEYS (SYNTHETIC demo only; revoke with: crisk revoke-api-key) ===
analyst: $ANALYST
auditor: $AUDITOR
UI: http://localhost:8501   API docs: http://localhost:8000/docs
KEYS
cat "artifacts/$DS/DEMO_KEYS.txt"
