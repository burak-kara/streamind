---
name: add-node
description: Scaffold a new Juturna node under plugins/nodes/<type>/_<name>/ with correct structure
---

Ask the user for:
- Node name (e.g. `my_processor`)
- Node type: `source`, `proc`, or `sink`

Then create the directory `plugins/nodes/<type>/_<name>/` with:

1. `__init__.py` — empty file

2. `<name>.py` — stub class inheriting the correct Juturna base:
   - `source` → inherit from `juturna.nodes.BaseSource`
   - `proc` → inherit from `juturna.nodes.BaseProcessor`
   - `sink` → inherit from `juturna.nodes.BaseSink`

   Stub must include `__init__` and the appropriate abstract method (`produce`, `process`, or `consume`).

3. After creation, remind the user to:
   - Register the node in `pipelines/config.json`
   - Add a corresponding test file in `tests/test_<name>.py`
