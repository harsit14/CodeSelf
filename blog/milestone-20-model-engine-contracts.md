# Milestone 20: Model Engine Contracts Before Model Training

This milestone is the second slice of Phase 4. The goal was to move one layer
closer to real RL training without making the clean smoke path depend on a
large machine-learning stack.

The repository now has a model-engine boundary, a tokenizer boundary, and a way
to pack generated code into the same training records that GRPO and PPO will
eventually consume.

## Tokenization as an explicit contract

The training core now has a `TokenizerEngine` protocol. It only requires
`encode`, `decode`, and an optional `eos_token_id`. That is small on purpose.
For unit tests, I added a deterministic `WhitespaceTokenizerEngine`; it is not a
real model tokenizer, but it gives us stable token ids, truncation behavior, and
decode round-trips without importing Transformers.

The important helper is `encode_prompt_response`. It encodes the prompt and
response separately, truncates them, concatenates the ids, and builds prompt and
response masks. By default, prompt truncation keeps the right side of the
prompt, while response truncation keeps the left side of the generated answer.
That matches the usual intuition: keep the newest task context and keep the
beginning of the program if a response is too long.

This is also where the MacBook M5 Pro with 48GB unified memory matters. Local
debugging should stay light enough to run constantly. I want the repo to test
contracts on the laptop, then reserve heavyweight model execution for explicit
integration runs.

## Generated sequences

I added `GenerationRequest` and `GeneratedSequence` records. A generated
sequence stores:

- the original prompt;
- the generated response text;
- packed `input_ids`;
- prompt, response, and attention masks;
- a finish reason such as `stop` or `length`;
- small metadata.

It can also turn itself into a `SequenceTrainingSample`. That means future
rollout code can generate text, execute it, receive a reward, and then convert
the result into the exact record shape expected by the training core.

This is the sort of interface that looks boring until it prevents three bugs at
once. The loss code should not have to guess where the prompt ends. The reward
code should not have to know how token masks are built. The model adapter should
not own the policy-gradient data model.

## Lazy Transformers adapter

There is now an optional `TransformersModelEngine`. It is intentionally lazy:
Torch and Transformers are imported only when the engine is instantiated. A
plain `import codeself.training` remains dependency-free.

The adapter can:

- load a causal language model and tokenizer from Hugging Face APIs;
- generate one response from a `GenerationRequest`;
- return token-aligned policy logprobs for a packed sequence.

The default config uses `local_files_only: true` so an integration test will not
silently download a model. That is the right bias for reproducible experiments:
model availability should be explicit.

## What this does not do yet

This milestone still does not attach LoRA adapters, run a GRPO loss, run a PPO
loss, or update model weights. It also does not prove that the Transformers path
works with a cached model on every target machine.

The next good step is either:

- a slow integration test with a tiny cached causal LM; or
- the first dependency-free GRPO loss helper that consumes the masks and
  logprob records already defined.

I lean toward the loss helper next, because it would let the project validate
the math with toy tensors before involving local model loading.
