# Chat Template System

## Design

Uses an Elixir behaviour (`ExLLama.ChatTemplate`) with 18 implementations. Each template knows how to:
1. Format a list of chat messages into a prompt string the model expects
2. Extract and parse the model's response back into a structured `GenAI.ChatCompletion`

## Behaviour

```elixir
@callback support_list() :: {:ok, MapSet.t}
@callback to_context(thread, model, meta) :: {:ok, String.t}
@callback extract_response([{tokens, text}], model, meta) :: {:ok, GenAI.ChatCompletion.t}
```

## Template Selection

```elixir
# Explicit selection via options
ExLLama.chat_completion(model, thread, template: ExLLama.ChatTemplate.Llama3Instruct)

# Default: Zephyr
```

Selection logic in `pick_handler/2`: uses `meta[:template]` if provided, otherwise defaults to `ExLLama.ChatTemplate.Zephyr`.

## Implementations

| Module | Format | Typical Models |
|--------|--------|----------------|
| `Alpaca` | Alpaca instruction format | Alpaca-based fine-tunes |
| `AmberChat` | AmberChat format | LLM360 Amber |
| `ChatML` | `<\|im_start\|>` delimited | Many OpenAI-style models |
| `ChatQA` | ChatQA format | Nvidia ChatQA |
| `FalconInstruct` | Falcon instruction format | Falcon models |
| `GemmaInstruct` | Gemma turn-based format | Google Gemma |
| `GraniteInstruct` | Granite instruction format | IBM Granite |
| `LLama2Chat` | `[INST]` delimited | Llama 2 Chat |
| `Llama3Instruct` | `<\|start_header_id\|>` format | Llama 3 Instruct |
| `MistralInstruct` | `[INST]` with `<s>` wrapping | Mistral Instruct |
| `OpenChat` | OpenChat 3.5 format | OpenChat models |
| `Phi3` | Phi-3 format | Microsoft Phi-3 |
| `Phi3Small` | Phi-3 Small variant | Microsoft Phi-3 Small |
| `QwenInstruct` | Qwen instruction format | Alibaba Qwen |
| `Saiga` | Saiga format | Russian Saiga models |
| `SolarInstruct` | Solar instruction format | Upstage Solar |
| `Vicuna` | Vicuna conversation format | Vicuna models |
| `Zephyr` | `<\|system\|>` delimited (default) | Zephyr, HuggingFace models |

## Companion Resources

The `chat_templates/` directory at project root contains reference Jinja2 templates (`.jinja`) and generation configs (`.json`) for each format, useful for validation and cross-referencing with HuggingFace model cards.
