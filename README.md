# Fine-Tuning Part 2 — Low-Rank Adaptation and Instruction Tuning

[Part 1](https://github.com/VinayakMokashi/finetuning-part1) fine-tuned *every* weight in a
pretrained model. This repository asks the obvious follow-up question: do you have to?

Three scripts answer it, and they are meant to be read in order. The first is pure linear
algebra and proves that a large weight matrix can sometimes be replaced by two very small
ones with no loss at all. The second uses exactly that fact to fine-tune BERT by training
0.27% of its parameters. The third steps sideways to a different kind of fine-tuning
altogether — teaching an encoder-decoder model to follow instructions.

The two training scripts follow the same shape as Part 1: **measure the pretrained model,
train it for one epoch, measure it again**, and print the difference. Each is written as a
single class so that every stage — model, data, loss, training arguments, trainer,
evaluation — is one named method you can read in isolation.

| Script | Task | Base model | Dataset | What it teaches |
| --- | --- | --- | --- | --- |
| [`finetuning_low_rank.py`](finetuning_low_rank.py) | Low-rank matrix approximation | — (NumPy only) | synthetic | SVD, rank truncation, and why LoRA's `ΔW = B·A` is not an approximation hack |
| [`finetuning03.py`](finetuning03.py) | Binary sentiment classification | [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) | [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) | LoRA with `peft`, adapter injection, cross-entropy implemented from scratch in NumPy |
| [`finetuning04.py`](finetuning04.py) | Instruction following | [`t5-small`](https://huggingface.co/t5-small) | [`finetune_instruction_data.csv`](finetune_instruction_data.csv) (12 rows, in-repo) | Seq2seq preprocessing, `-100` label masking, and encoder-decoder loss |

Like Part 1, these are deliberately sized to finish on a laptop CPU — 500 training reviews
and 10 instruction examples — not to produce competitive models.

---

## Quick start

```bash
git clone https://github.com/VinayakMokashi/finetuning-part2.git
cd finetuning-part2

python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

python finetuning_low_rank.py   # instant, no downloads
python finetuning03.py          # ~20 min on a laptop CPU
python finetuning04.py          # ~1 min on a laptop CPU
```

`finetuning_low_rank.py` needs nothing but NumPy and finishes in under a second. The other
two download their model and dataset from the Hugging Face Hub on first run — 421 MB for
BERT, 233 MB for T5 and 80 MB for IMDb, about 750 MB in total, cached under
`~/.cache/huggingface`. No account, token or API key is required.

**Requirements:** Python 3.10 or newer — verified on 3.12. Budget roughly **2 GB of free
disk space**: ~750 MB of downloads plus training checkpoints. A GPU is optional — see
[Running on a GPU](#running-on-a-gpu).

---

## `finetuning_low_rank.py` — why LoRA is allowed to work

### The claim

LoRA freezes a pretrained weight matrix `W` and learns an update in factored form,
`ΔW = B·A`, where `A` and `B` are far smaller than `W`. That only makes sense if the update
you need is genuinely low-rank. This script builds a matrix that *is*, and shows the
factorisation recovering it exactly.

### How it works

The matrix is low-rank by construction — a 1000×1000 matrix built from a 1000×2 times a
2×1000, so all thousand of its rows really live in a 2-dimensional subspace:

```python
W = np.random.randn(1000, rank) @ np.random.randn(rank, 1000)   # rank = 2
```

The SVD then *finds* that structure without being told it is there. `np.linalg.svd` factors
`W = U · diag(S) · Vᵀ`, where the singular values in `S` measure how much signal each
direction carries. Note the `V = V.T` on the following line: NumPy returns `Vᵀ`, so this
flips it back so that the columns of `V` are the right singular vectors.

Keeping only the top `estimated_rank` columns and repackaging them gives LoRA's two
matrices — `A` projects 1000 dimensions down to `r`, and `B` projects back up:

```python
A = V_r.T          # r × 1000   (down-projection)
B = U_r @ S_r      # 1000 × r   (up-projection)
W_hat = B @ A
```

### What to expect

```
W shape: (1000, 1000)
params in W: 1000000
params in A + B: 4000
top singular values: [9.77386710e+02 9.67106779e+02 1.17437705e-12 9.94510915e-13 8.68093783e-13]
reconstruction error ||W - W_hat||: 1.081446513076062e-12
output error ||y - y_hat||: 1.0639366357095323e-12
```

Two large singular values and then a cliff to ~1e-12 — the SVD has correctly identified the
rank. The reconstruction stores **250× fewer parameters** and reproduces `y = Wx + b` to
floating-point precision. (The matrices are random, so your singular values will differ; the
cliff after the second one will not.)

The instructive experiment is to change `estimated_rank`:

| `estimated_rank` | Reconstruction error | Reading |
| --- | --- | --- |
| 1 | ≈ 967 | Under-ranked — a real singular direction was discarded. This is what an undersized LoRA `r` feels like in practice. |
| 2 | ≈ 1e-12 | Exact. The smallest rank that loses nothing. |
| 3 | ≈ 1e-12 | Also exact, but the third direction carries no signal — wasted parameters. |

Real weight updates are never *exactly* low-rank, which is why choosing `r` in a real LoRA
run is an empirical trade-off rather than a clean cliff. But the mechanism is the one above.

---

## `finetuning03.py` — LoRA fine-tuning for sentiment classification

### The task

The same task as `finetuning02.py` in Part 1 — binary sentiment on IMDb movie reviews with
`bert-base-uncased`, on the same 500-review training subset — but trained with LoRA instead
of updating every weight.

Two training settings differ from Part 1 deliberately. The evaluation subset is 100 reviews
rather than 32, and the learning rate is `1e-4` rather than `2e-5`. That second change is
the standard adjustment when moving to LoRA: the adapters are randomly initialised and few
in number, so they tolerate — and need — a learning rate roughly an order of magnitude
larger than the one you would use to nudge pretrained weights.

### How it works

`LoraConfig` describes the adapters, and `get_peft_model` injects them into the frozen base
model:

```python
LoraConfig(
    peft_type=PeftType.LORA,
    task_type="SEQ_CLS",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
```

- `r=8` is the rank of the `B·A` factorisation from the previous script — the width of the
  bottleneck each adapter squeezes through.
- `lora_alpha=32` scales the adapter output by `alpha / r = 4`. It exists so that changing
  `r` does not silently change the effective size of the adapter's contribution.
- `task_type="SEQ_CLS"` matters for more than bookkeeping: it tells `peft` that the freshly
  initialised classification head must stay trainable. That head is randomly initialised by
  `AutoModelForSequenceClassification` and would be useless if frozen along with the rest.

The result, which the script prints at startup:

```
trainable params: 296,450 || all params: 109,780,228 || trainable%: 0.2700
```

**0.27% of the model is being trained.** Everything else is frozen, so the optimiser state
and gradient buffers shrink by roughly the same factor — which is the practical reason LoRA
is used at all.

### Cross-entropy from scratch

As in Part 1, `evaluate_loss` does not call the framework's loss. It runs the model, pulls
the logits out to NumPy, and computes cross-entropy by hand in `cross_entropy_loss` —
including the standard numerical-stability trick of subtracting the row maximum before
exponentiating:

```python
shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
```

Without it, a large logit overflows `np.exp` to `inf` and the softmax returns `nan`.
Subtracting the maximum leaves the softmax mathematically unchanged.

This implementation is verified rather than assumed. Because there is exactly one label per
example, the per-example average it computes is the same quantity the `Trainer` reports, and
on the full run below the two agreed to every digit the `Trainer` prints: `0.6816` against
`0.6816`.

### What to expect

```
trainable params: 296,450 || all params: 109,780,228 || trainable%: 0.2700
{'eval_loss': '0.6816', 'eval_runtime': '42.92', 'epoch': '1'}
{'train_runtime': '1026', 'train_loss': '0.6772', 'epoch': '1'}
eval loss before fine-tuning: 0.6862
eval loss after  fine-tuning: 0.6816
improvement: 0.0045
```

The starting value sits just under `ln(2) ≈ 0.6931`, which is exactly where a randomly
initialised 2-way head belongs — the model begins with no opinion about sentiment. After one
epoch the loss has moved, but barely. **That small number is the honest result, and it is
worth sitting with rather than explaining away** — see
[Verified results](#verified-results) below for what it does and does not tell you.

---

## `finetuning04.py` — instruction fine-tuning a seq2seq model

### The task

The first two scripts classify. This one *generates*. `t5-small` is an encoder-decoder model,
and the goal is to teach it to follow written instructions: given an instruction and an
input, produce the requested output.

The dataset is [`finetune_instruction_data.csv`](finetune_instruction_data.csv), twelve
hand-written examples covering summarisation, translation, sentiment classification,
grammar correction, paraphrasing and more. Ten rows train, two evaluate. `load_dataset_dict`
is kept in the script as the in-code source of that same data, so you can read the examples
without opening the CSV.

### Prompt construction

Each row is flattened into a single string that the encoder reads:

```python
f"Instruction: {inst}\nInput: {inp}"
```

The target is the expected output. Nothing about this format is magic — what matters is that
it is *consistent*, so the model can learn that the text after `Instruction:` describes the
transformation to apply to the text after `Input:`.

### Label masking, and why it matters

Targets are padded to a fixed length, and those pad positions must not be treated as tokens
the model was supposed to predict. Every pad in the labels is therefore replaced with `-100`,
PyTorch's conventional ignore index:

```python
model_inputs["labels"] = [
    [token if token != self.tokenizer.pad_token_id else -100 for token in seq]
    for seq in labels
]
```

Skip this and the loss is dominated by the trivial task of predicting padding — the number
goes down, the model does not get better. Two pieces of machinery then rely on the
convention:

- `torch.nn.CrossEntropyLoss`, used internally by T5, ignores `-100` positions outright, and
  the NumPy loss in `evaluate_loss` reproduces that with `valid_indices = shift_labels != -100`.
- T5's `_shift_right`, which builds `decoder_input_ids` by shifting the labels one position
  right, replaces any `-100` with the pad token before it is embedded. Without that step the
  embedding lookup would be handed a negative index.

That second point is what makes the manual evaluation loop safe:

```python
decoder_input_ids = model._shift_right(labels)
```

Teacher forcing, in one line: at every step the decoder is shown the *correct* previous
token rather than its own last guess, so a whole sequence can be scored in a single forward
pass instead of a generation loop.

### A note on the two loss numbers

The script prints its own NumPy loss, while the `Trainer` prints `eval_loss` during training.
They will not match exactly, and the reason is worth understanding: `evaluate_loss` averages
per **example** (a macro average), whereas the `Trainer` averages per **token** (a micro
average). Examples with longer targets carry more weight in the second.

This was checked directly. On one run, the two evaluation rows scored 3.9070 and 3.7611 over
11 and 4 real target tokens respectively. Averaging them per example gives 3.8340; weighting
them by token count gives 3.8681 — and the `Trainer` reported `eval_loss` of 3.868 on that
same run. Neither number is wrong; they answer slightly different questions.

Those figures come from a run that skipped the pre-training measurement, so they do not line
up with the ones in the next section. That is not noise — see
[Reproducibility](#reproducibility).

### What to expect

```
eval loss before fine-tuning: 3.9626
eval loss after  fine-tuning: 3.8433
improvement: 0.1193
```

**Read this script for its mechanics, not its results.** Ten training examples and one epoch
cannot teach instruction following — real instruction tuning uses tens of thousands of
examples at minimum. What the loss drop confirms is that the pipeline is wired correctly and
gradients are flowing to the right places. To make it a genuine experiment, point
`load_dataset` at a real instruction dataset such as
[`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) and raise
`num_train_epochs`.

---

## Verified results

Measured on the reference machine below, running each script exactly as committed.

### `finetuning03.py` — LoRA on IMDb

| Metric | Value |
| --- | --- |
| Trainable parameters | `296,450` of `109,780,228` (**0.27%**) |
| Eval cross-entropy, **before** fine-tuning | `0.6862` |
| Eval cross-entropy, **after** fine-tuning | `0.6816` |
| Change | **-0.0045** |
| Final training loss | `0.6772` |
| Training wall-clock (63 steps) | ~17 min |

### `finetuning04.py` — instruction tuning t5-small

| Metric | Value |
| --- | --- |
| Eval loss, **before** fine-tuning | `3.9626` |
| Eval loss, **after** fine-tuning | `3.8433` |
| Change | **-0.1193** |
| Final training loss | `4.055` |
| Training wall-clock (2 steps) | ~17 s |

### Reading the LoRA number honestly

Part 1 fine-tuned *the same model on the same 500 IMDb reviews* and moved the loss from
`0.6809` to `0.3024` — a change of `-0.3784`, roughly eighty times larger than the `-0.0045`
here. It would be easy to read that as "LoRA is much worse". It is not that simple, and
three things differ at once:

1. **Optimiser steps: 63 versus 500.** Part 1 used `per_device_train_batch_size=1`, so 500
   examples meant 500 gradient updates. This script uses batch size 8, so the same data
   yields 63. That is an eight-fold reduction in updates and is very likely the largest
   single factor.
2. **Trainable parameters: 0.27% versus 100%.** Fewer degrees of freedom means less can
   change per step, by design.
3. **The classification head.** In both cases it starts random, but here it is being learned
   through far fewer updates.

The published LoRA result is that it *approaches* full fine-tuning given a comparable
training budget — not that it matches it after 63 steps. Nothing in this repository
establishes the comparison either way, because the budgets were never equalised. If you want
to test it properly, set `per_device_train_batch_size=1` in `setup_training_args` to match
Part 1 step for step, or raise `num_train_epochs` until the update counts line up.

What this run *does* establish is the parameter accounting: a 110M-parameter model was
adapted by training 296,450 weights, and the loss moved in the right direction.

### Reproducibility

These scripts are more deterministic than they look. Neither sets a seed explicitly, but
`TrainingArguments` defaults to `seed=42` and `Trainer.__init__` calls `set_seed`, so
shuffling and dropout *during training* are seeded. Running `python finetuning04.py` three
times on the reference machine printed `3.9626 / 3.8433 / 0.1193` every time.

Two things still move between runs, and both are questions of ordering rather than luck:

- **The starting loss in `finetuning03.py`.** `AutoModelForSequenceClassification` builds the
  randomly initialised classification head inside `setup_base_model()`, several steps before
  the `Trainer` is constructed and seeds the RNG. That head is different in every process, so
  the "before" figure lands somewhere new each time. The *direction* of the change is the
  reproducible result, not the starting value.
- **Measuring the loss before training changes the training run.** PyTorch's `DataLoader`
  draws a base seed from the global RNG when its iterator is created — even with
  `shuffle=False` — so the pre-training evaluation advances the generator and dropout draws
  differently from then on. This is deterministic rather than random: skipping the "before"
  measurement in `finetuning04.py` reproducibly gives an `eval_loss` of `3.868` where the
  committed script gives `3.891`.

Calling `transformers.set_seed(42)` at the top of `__main__`, before the model is built,
pins both down if you need exact repeatability.

`finetuning_low_rank.py` seeds nothing and is genuinely random on each run — but the cliff in
the singular values after the second one appears every time, which is the only part of that
output the script is trying to show.

### Reference machine

Windows 11, Python 3.12.3, **CPU only** (no CUDA), torch 2.12.1+cpu, transformers 5.12.1,
peft 0.20.0, datasets 5.0.0, accelerate 1.14.0, numpy 2.2.6, pandas 3.0.3.

---

## Running on a GPU

Nothing in these scripts is CPU-specific. `Trainer` moves the model and batches to CUDA
automatically when a GPU is visible, so the only change needed is a CUDA build of PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Once a GPU is present it is worth raising `per_device_train_batch_size` in
`setup_training_args`, and enabling mixed precision with `fp16=True` (or `bf16=True` on
Ampere and newer).

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'peft'`** — `finetuning03.py` needs it; run
`pip install -r requirements.txt` inside the activated virtual environment.

**`HfUriError: Repository id must be 'namespace/name'`** — recent versions of `datasets` and
`huggingface-hub` no longer resolve bare dataset names such as `"imdb"`. This repository uses
the fully qualified `"stanfordnlp/imdb"` throughout.

**Checkpoint directories appear somewhere unexpected** — `Trainer` resolves `output_dir`
against the current working directory, so `peft_results/` and `instruction_result/` are
created wherever you launch Python from. The dataset CSV, by contrast, is resolved against
the script's own location, so `finetuning04.py` itself runs correctly from any directory.

**The first run is slow or appears to hang** — it is downloading model weights and the IMDb
dataset. Subsequent runs read from the Hugging Face cache and start immediately.

---

## Repository layout

```
finetuning_low_rank.py            SVD and low-rank approximation — the maths behind LoRA
finetuning03.py                   LoRA fine-tuning: BERT + IMDb via peft
finetuning04.py                   Instruction fine-tuning: t5-small on a small CSV dataset
finetune_instruction_data.csv     12 instruction / input / output rows
requirements.txt                  Pinned lower bounds for the whole stack
```

Training checkpoints are written to `./peft_results` and `./instruction_result`, both
git-ignored — they are large and reproducible by re-running the scripts.

---

## License

Released under the [MIT License](LICENSE).
