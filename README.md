# Fine-Tuning Part 2 — Low-Rank Adaptation and Instruction Tuning

[Part 1](https://github.com/VinayakMokashi/finetuning-part1) fine-tuned *every* weight in a
pretrained model. This repository asks the obvious follow-up question: do you have to?

Three scripts answer it, and they are meant to be read in order. The first is pure linear
algebra and proves that a large weight matrix can sometimes be replaced by two very small
ones with no loss at all. The second uses exactly that fact to fine-tune BERT by training
0.27% of its parameters, reaching 80% accuracy on a task it started at chance. The third
steps sideways to a different kind of fine-tuning altogether — teaching an encoder-decoder
model to follow written instructions, where the evidence is not a loss curve but what the
model actually writes.

The two training scripts follow the same shape as Part 1: **measure the pretrained model,
train it for one epoch, measure it again**, and print the difference. Each is written as a
single class so that every stage — model, data, loss, training arguments, trainer,
evaluation — is one named method you can read in isolation.

| Script | Task | Base model | Dataset | What it teaches |
| --- | --- | --- | --- | --- |
| [`finetuning_low_rank.py`](finetuning_low_rank.py) | Low-rank matrix approximation | — (NumPy only) | synthetic | SVD, rank truncation, and why LoRA's `ΔW = B·A` is not an approximation hack |
| [`finetuning03.py`](finetuning03.py) | Binary sentiment classification | [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) | [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) | LoRA with `peft`, adapter injection, cross-entropy implemented from scratch in NumPy |
| [`finetuning04.py`](finetuning04.py) | Instruction following | [`t5-small`](https://huggingface.co/t5-small) | [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) (2,000 rows) | Seq2seq preprocessing, `-100` label masking, held-out generation, and encoder-decoder loss |

Like Part 1, these are deliberately sized to finish on a laptop CPU — 500 training reviews
and 2,000 instruction examples — not to produce competitive models.

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
python finetuning03.py          # ~9 min on a laptop CPU
python finetuning04.py          # ~25 min on a laptop CPU
```

`finetuning_low_rank.py` needs nothing but NumPy and finishes in under a second. The other
two download their models and datasets from the Hugging Face Hub on first run — 421 MB for
BERT, 233 MB for T5, 80 MB for IMDb and 24 MB for Alpaca, about 760 MB in total, cached
under `~/.cache/huggingface`. No account, token or API key is required.

**Requirements:** Python 3.10 or newer — verified on 3.12. Budget roughly **1 GB of free
disk space**; neither script writes checkpoints, so the downloads are nearly all of it. A GPU
is optional — see [Running on a GPU](#running-on-a-gpu).

In a hurry? `finetuning04.py` has a twelve-row CSV built in for exactly that — see
[the smoke test](#a-note-on-the-twelve-row-csv).

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
| 1 | ≈ 900–1000 | Under-ranked — a real singular direction was discarded. This is what an undersized LoRA `r` feels like in practice. |
| 2 | ≈ 1e-12 | Exact. The smallest rank that loses nothing. |
| 3 | ≈ 1e-12 | Also exact, but the third direction carries no signal — wasted parameters. |

The error at `estimated_rank = 1` is not an arbitrary number: it equals the discarded second
singular value, to every digit. That is the Eckart–Young theorem — truncating an SVD is
provably the *best* approximation of that rank, and the error it leaves behind is exactly
the energy you threw away. Rounded values differ between runs only because the matrices are
freshly random each time.

Real weight updates are never *exactly* low-rank, which is why choosing `r` in a real LoRA
run is an empirical trade-off rather than a clean cliff. But the mechanism is the one above.

---

## `finetuning03.py` — LoRA fine-tuning for sentiment classification

### The task

The same task as `finetuning02.py` in Part 1 — binary sentiment on IMDb movie reviews with
`bert-base-uncased`, on the same 500-review training subset — but trained with LoRA instead
of updating every weight.

The evaluation subset is 100 reviews rather than Part 1's 32, and the learning rate is
`5e-4` rather than `2e-5`. That second difference is the standard adjustment when moving to
LoRA: the adapters start from scratch and there are very few of them, so they tolerate — and
need — a learning rate more than an order of magnitude larger than the one you would use to
nudge pretrained weights. Getting this wrong is not a subtle failure, as the
[training budget](#the-training-budget-matters-more-than-it-looks) section below shows.

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
on the run below the two agreed to every digit the `Trainer` prints: `0.4661` against
`0.4661`.

### The training budget matters more than it looks

An earlier version of this script trained with a learning rate of `1e-4` and a batch size of
8, and produced this:

```
eval loss before fine-tuning: 0.6862
eval loss after  fine-tuning: 0.6816
```

A two-way classifier with no opinion whatsoever scores `ln(2) ≈ 0.6931`. That run therefore
started at chance and finished at chance. The adapters were configured correctly the whole
time — they were simply never pushed hard enough to learn anything. Three things fixed it:

- **More updates.** 500 reviews at batch size 8 is 63 optimiser steps. Dropping the batch
  size to 4 doubles that to 125 for exactly the same data and roughly the same wall clock.
- **A much higher learning rate**, `5e-4` with 10% warmup, for the reason given above.
- **Dynamic padding.** Every review used to be padded to 512 tokens regardless of length.
  Padding each batch to its own longest sequence instead, capped at 256 tokens, is where the
  compute for the extra updates came from — the run got *faster*, from 17 minutes to 7.

The lesson generalises: when a parameter-efficient method appears not to work, suspect the
training budget before the method.

### One epoch is deliberate

Running three epochs instead of one is worse, and the script's own per-epoch evaluation says
so:

| Epoch | Training loss | Eval loss | Eval accuracy |
| --- | --- | --- | --- |
| 1 | 0.4433 | **0.3858** | 0.88 |
| 2 | 0.2536 | 0.5830 | 0.85 |
| 3 | 0.2047 | 0.5669 | 0.88 |

Training loss keeps falling while evaluation loss climbs after epoch 1 — the model is
memorising 500 reviews rather than learning sentiment. Epochs 2 and 3 cost fourteen minutes
and bought nothing. This is a within-run comparison, so it is not an artefact of
initialisation luck.

### What to expect

```
trainable params: 296,450 || all params: 109,780,228 || trainable%: 0.2700
{'eval_loss': '0.4661', 'eval_accuracy': '0.8', 'epoch': '1'}
{'train_runtime': '440.7', 'train_loss': '0.6039', 'epoch': '1'}
accuracy  before fine-tuning: 0.5100
accuracy  after  fine-tuning: 0.8000
eval loss before fine-tuning: 0.6964
eval loss after  fine-tuning: 0.4661
loss improvement: 0.2303
```

The starting point is the interesting part. Accuracy of `0.51` on a balanced two-way task is
a coin flip, and the loss of `0.6964` sits right at `ln(2) ≈ 0.6931` — exactly where a
randomly initialised head belongs. The model genuinely begins with no opinion about
sentiment.

After 125 optimiser updates, training 0.27% of its weights, it gets **four reviews in five
right**.

---

## `finetuning04.py` — instruction fine-tuning a seq2seq model

### The task

The first two scripts classify. This one *generates*. `t5-small` is an encoder-decoder model,
and the goal is to teach it to follow written instructions: given an instruction and an
input, produce the requested output.

The dataset is [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca), 52,002
instruction / input / output rows. The script shuffles once with a fixed seed and takes 2,000
rows to train on and a further 200, **never trained on**, to evaluate. Holding out an
evaluation split matters more here than for a classifier: a generative model that has already
seen its test prompts will happily recite the answers back.

### A note on the twelve-row CSV

This script began with [`finetune_instruction_data.csv`](finetune_instruction_data.csv) —
twelve hand-written rows covering summarisation, translation, sentiment classification,
grammar correction and paraphrasing. It is still in the repository, and `load_dataset_csv()`
still reads it, because it makes a genuinely useful smoke test: point `load_dataset()` at it
to exercise the entire pipeline in seconds, offline.

What it cannot do is teach instruction following. Ten training rows is not a small dataset,
it is a *demonstration* — and training on it for more epochs does not fix that, it only
memorises ten answers. The loss falls impressively while the model gets worse at everything
it has not seen. Real instruction tuning starts in the tens of thousands of examples, which
is why the default moved to Alpaca.

### Prompt construction

Each row is flattened into a single string that the encoder reads:

```python
f"Instruction: {instruction}\nInput: {input_text}"
```

The target is the expected output. Nothing about this format is magic — what matters is that
it is *consistent*, so the model can learn that the text after `Instruction:` describes the
transformation to apply to the text after `Input:`.

Roughly half of Alpaca's rows have an empty `input` field — "Give three tips for staying
healthy" needs no operand. Those rows drop the `Input:` line entirely rather than emitting a
dangling empty one, so the model never has to make sense of a trailing `Input:` with nothing
after it.

### Label masking, and why it matters

Batches are padded to a common length, and those pad positions must not be treated as tokens
the model was supposed to predict. Every pad in the labels is therefore replaced with `-100`,
PyTorch's conventional ignore index. `DataCollatorForSeq2Seq` does this while it pads:

```python
DataCollatorForSeq2Seq(
    tokenizer=self.tokenizer,
    model=self.model,
    label_pad_token_id=-100,
)
```

Skip this — pad the labels with the tokeniser's ordinary pad token, as an earlier version of
this script did — and the loss is dominated by the trivial task of predicting padding. The
number goes down and the model does not get better. Two pieces of machinery rely on the
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

Measured on the untrained model over the 200 held-out rows:

| Quantity | Value |
| --- | --- |
| `evaluate_loss()`, the from-scratch NumPy version | `3.9710` |
| The same batches scored by PyTorch's own `CrossEntropyLoss` | `3.9711` |
| The same batches, weighted by token count instead | `3.9418` |
| `Trainer.evaluate()` | `3.9582` |

The first two lines are the verification: **the hand-written NumPy cross-entropy agrees with
PyTorch's own implementation to four decimal places** on identical batches.

The remaining spread is not error, it is normalisation. Cross-entropy is averaged over the
real tokens *within each batch*, and then the batches are averaged together. That makes the
final number depend on how examples were grouped: `evaluate_loss` uses batches of 4 while
`Trainer` uses 8, which is the whole of the difference between `3.9710` and `3.9582`.
Weighting every token equally across the entire split instead gives a third answer, `3.9418`.

None of the three is wrong. It is worth knowing that a reported loss carries a batch size
with it, and that comparing losses comes with a quiet assumption that the batching matched.

### What to expect

```
eval loss before fine-tuning: 3.9710
eval loss after  fine-tuning: 2.9116
improvement: 1.0594
```

The loss is the smaller half of the story. The script also prints what the model actually
writes, on prompts it was never trained on — and that is where the change is legible.

**Before**, `t5-small` does not follow instructions at all. It mostly echoes the prompt back
verbatim:

```
prompt    : Instruction: Create a set of guidelines for businesses to follow in order to build customer trust.
generated : Instruction: Create a set of guidelines for businesses to follow in order to build customer trust.
```

Occasionally something stranger surfaces — this prompt made it start translating into German:

```
prompt    : Instruction: A new restaurant has opened up in town. Come up with six menu items ...
generated : Instruction: Inserieren Sie sich auf einen neuen Restaurant in Town. Come up with six menu items ...
```

That is not a bug, it is T5's pretraining showing through. T5 was trained on a mixture of
tasks introduced by short text prefixes, translation among them, so a sentence beginning
`Instruction:` lands in a familiar-looking but wrong place.

**After** two epochs on 2,000 examples, it attempts the task:

```
prompt    : Instruction: Create a set of guidelines for businesses to follow in order to build customer trust.
generated : 1. Establish a strong customer relationship with a customer. 2. Establish a strong customer relationship with a customer. 2. Establish a strong customer relatio

prompt    : Instruction: Create a dialogue between two people that incorporates the given ideas.
            Input: Ideas: money saving tips, weekly budget
generated : The idea of money saving tips is to create a weekly budget. It is a way to get started by focusing on the basics of the budget, focusing on the budget, and focu
```

**This is progress, not success, and the difference matters.** The model has learned the
*shape* of the task — it stops echoing, produces a numbered list when asked for guidelines,
and picks up "money saving tips" and "weekly budget" from the input. It has not learned to
write well: it falls into repetition loops, and the second answer is a description of a
dialogue rather than a dialogue.

Both failures are expected here. `t5-small` is 60M parameters, greedy decoding has no
repetition penalty, and 2,000 examples is a rounding error next to the 52,002 available. The
honest summary is that the pipeline demonstrably teaches instruction *following*, and that
teaching instruction following *well* is a different-sized problem. Raise `train_size` and
`num_train_epochs`, or move to `t5-base`, to push it further.

---

## Verified results

Measured on the reference machine below, running each script exactly as committed.

### `finetuning03.py` — LoRA on IMDb

| Metric | Value |
| --- | --- |
| Trainable parameters | `296,450` of `109,780,228` (**0.27%**) |
| Eval accuracy, **before** fine-tuning | `0.5100` |
| Eval accuracy, **after** fine-tuning | **`0.8000`** |
| Eval cross-entropy, **before** fine-tuning | `0.6964` |
| Eval cross-entropy, **after** fine-tuning | `0.4661` |
| Loss change | **-0.2303** |
| Final training loss | `0.6039` |
| Training wall-clock (125 steps) | ~7 min 20 s |

### `finetuning04.py` — instruction tuning t5-small

Evaluated on 200 Alpaca rows that were never trained on.

| Metric | Value |
| --- | --- |
| Eval loss, **before** fine-tuning | `3.9710` |
| Eval loss, **after** fine-tuning | **`2.9116`** |
| Change | **-1.0594** |
| Eval loss after epoch 1 / epoch 2 | `2.944` / `2.909` |
| Final training loss | `3.155` |
| Training wall-clock (500 steps) | ~14 min |

Unlike `finetuning03.py`, this one is **not** overfitting — evaluation loss was still falling
between epoch 1 and epoch 2. 2,000 examples is enough data that two epochs does not exhaust
it, so raising `num_train_epochs` here is a reasonable thing to try, where in the LoRA script
it was actively harmful.

### Comparing against Part 1, carefully

Part 1 fine-tuned *the same model on the same 500 IMDb reviews*, updating every weight, and
reached a loss of `0.3024`. This script reaches `0.4661`. Full fine-tuning still wins on
loss — but it is worth being precise about what each side spent:

| | Part 1 (full) | Here (LoRA) |
| --- | --- | --- |
| Trainable parameters | 109,780,228 | **296,450** |
| Optimiser updates | 500 | 125 |
| Eval cross-entropy | **0.3024** | 0.4661 |
| Training wall-clock | ~25 min | **~7 min** |

LoRA gets within striking distance of full fine-tuning while training **370× fewer
parameters**, in **under a third of the wall-clock time and a quarter of the updates**. That is
the trade LoRA actually offers, and it is visible here.

What this still does *not* establish is which method wins at equal budget, because the
budgets are not equal — 125 updates against 500. The published result is that LoRA
approaches full fine-tuning given comparable training; testing that properly here would mean
setting `per_device_train_batch_size=1` to match Part 1 step for step. Note also that Part 1
reports no accuracy, so the `0.80` figure above has nothing to compare against.

One caveat on precision: the evaluation set is 100 reviews, so a single percentage point is
one review. Treat `0.80` as "roughly four in five", not as a measurement good to two digits.

### Reproducibility

Both training scripts call `set_seed(42)` as the very first thing they do, and the placement
is the point.

`TrainingArguments` already defaults to `seed=42`, and `Trainer.__init__` calls `set_seed`
on your behalf — so it is easy to assume the job is done. It is not, because `Trainer` is
constructed *last*. In `finetuning03.py` the classification head and the LoRA adapters are
initialised several steps earlier, in `setup_base_model()` and `setup_peft_model()`, while
the RNG is still wherever the process happened to leave it. Everything the `Trainer` seeds is
downstream of a starting point that was never seeded at all.

That is not a theoretical concern. Final accuracy landed at `0.88`, `0.79` and `0.80` across
three runs during development — a nine-point spread on a 100-review evaluation set. Those
runs were not a controlled seed experiment (they differed in epoch count as well), so the
spread cannot be attributed to initialisation alone. But it is wide enough to make the point:
while the starting weights are redrawn on every run, a difference between two hyperparameter
settings and a difference between two lucky draws are indistinguishable. Two processes now
produce byte-identical initial weights, verified directly.

One further ordering effect is worth knowing about, because it survives seeding and is
deterministic rather than random: PyTorch's `DataLoader` draws a base seed from the global
RNG when its iterator is created, *even with* `shuffle=False`. Measuring the loss before
training therefore advances the generator and changes the dropout draws during training. It
is reproducible — the same code takes the same path every time — but it means "evaluate then
train" and "train only" are genuinely different experiments.

`finetuning_low_rank.py` seeds nothing and is random on each run by design. The cliff in the
singular values after the second one appears every time, which is the only part of that
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

Once a GPU is present, enable mixed precision with `fp16=True` (or `bf16=True` on Ampere and
newer). Raising `per_device_train_batch_size` is the usual next step, but do it deliberately
in `finetuning03.py`: batch size there is what sets the number of optimiser updates, and
too few updates is precisely what made the original version of that script fail to learn.
Doubling the batch halves the updates, so raise `num_train_epochs` to compensate — and
re-check the per-epoch evaluation loss, because more epochs overfit 500 reviews quickly.

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
or Alpaca dataset. Subsequent runs read from the Hugging Face cache and start immediately.

**You need to run offline, or want a result in seconds** — `finetuning04.py` ships with a
twelve-row CSV for exactly this. Point `load_dataset()` at `load_dataset_csv()` and the whole
pipeline runs without touching the network. It demonstrates the mechanics; it does not train
a useful model. See [the note on the CSV](#a-note-on-the-twelve-row-csv).

**`finetuning04.py` repeats itself** — output like "Establish a strong customer relationship.
2. Establish a strong customer relationship." is expected. Generation is greedy with no
repetition penalty, and `t5-small` is small. Pass `repetition_penalty` or `no_repeat_ngram_size`
to `generate()`, or train on more data, to reduce it.

---

## Repository layout

```
finetuning_low_rank.py            SVD and low-rank approximation — the maths behind LoRA
finetuning03.py                   LoRA fine-tuning: BERT + IMDb via peft
finetuning04.py                   Instruction fine-tuning: t5-small on a small CSV dataset
finetune_instruction_data.csv     12 instruction / input / output rows
requirements.txt                  Pinned lower bounds for the whole stack
```

`./peft_results` and `./instruction_result` are the `Trainer` output directories. Both
scripts set `save_strategy="no"`, so no checkpoints are written and the directories stay
empty; they are git-ignored regardless. Raise `save_strategy` if you want to keep a
fine-tuned model, and expect roughly 1.3 GB per saved BERT checkpoint.

---

## License

Released under the [MIT License](LICENSE).
