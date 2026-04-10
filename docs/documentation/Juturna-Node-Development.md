# Juturna Node Development

## Node File Structure

```text
plugins/nodes/<type>/_<node_name>/
├── <node_name>.py       # Class named in CamelCase
├── config.toml          # Default args (overridden by pipeline config.json)
├── requirements.txt
└── readme.md
```

Node folder must start with `_`. Python file name = folder name without `_`.

## config.toml Format

```toml
[arguments]
model = "small.en"
device = "auto"
threshold = 0.5

[meta]
```

Arguments become constructor parameters; types inferred from defaults. Pipeline `configuration` JSON overrides these defaults.

## Node Class Template

```python
from juturna.components import Node, Message
from juturna.payloads import BasePayload  # or AudioPayload, ObjectPayload, etc.

class MyNode(Node[InputPayload, OutputPayload]):
    def __init__(self, arg1: str, arg2: int, **kwargs):
        super().__init__(**kwargs)

    def configure(self): pass   # acquire ports/connections (runs before warmup)
    def warmup(self): pass      # load models/state
    def start(self): super().start()    # required
    def stop(self): super().stop()      # required
    def destroy(self): pass             # cleanup

    def update(self, message: Message[InputPayload]):
        out = Message(creator=self.name, version=message.version,
                      payload=OutputPayload(...), timers_from=message)
        self.transmit(out)
```

## Payload Types

| Type | Contents |
|------|----------|
| `AudioPayload` | `np.ndarray` waveform + `sample_rate`, `channels`, timestamps |
| `ImagePayload` | `np.ndarray` + `width`, `height`, `depth`, `pixel_format`, `timestamp` |
| `VideoPayload` | List of `ImagePayload` + `fps`, `duration` |
| `BytesPayload` | Raw bytes |
| `ObjectPayload` | `dict` subclass — use for structured text/JSON outputs |
| `Batch` | List of messages — produced when multi-input synchronization fires |

**Immutability**: received messages and all payloads are immutable. Use `Draft(PayloadType)` for iterative construction; it freezes automatically on `transmit()`. Always create a new `Message` for output — never mutate the received one.

## Timing

```python
with message.timeit('step_name'):
    result = do_work()  # message.timers['step_name'] = elapsed seconds
# Propagate timers downstream:
out = Message(..., timers_from=message)
```

## Threading Model

Each node runs up to 3 threads:

- `_worker`: receives inbound messages into a buffer (decouples upstream from processing)
- `_update`: consumes buffer, calls `update()` — **only place user code runs**
- `_source` (source nodes only): repeatedly calls `_source_f()` to inject data

## Multi-input Synchronization

Default is passthrough (forward every message). For custom batching implement:

```python
def next_batch(self, sources: dict) -> dict:
    # sources: {upstream_node_name: [messages]}
    # return: {upstream_node_name: message_to_consume}
```

## Environment Variable Substitution

In `pipelines/config.json`, use `"$JT_ENV_MY_VAR"` anywhere in node `configuration` to inject env vars at runtime.

## Additional CLI Commands

```bash
uv run python -m juturna validate -c pipelines/config.json -p ./plugins --deep
uv run python -m juturna stub -n my_node -t proc -d ./plugins/nodes   # scaffold a node
uv run python -m juturna require -c pipelines/config.json -p ./plugins -s requirements.txt
uv run python -m juturna serve --port 8000 --folder ./pipelines        # HTTP manager
```
