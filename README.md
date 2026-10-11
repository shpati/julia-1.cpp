# julia-1.cpp decision engine in pure C

A dependency-free (libc + libm, optional OpenMP) C implementation of the inference
path of [`SupersonicLabs/Julia-1`](https://huggingface.co/SupersonicLabs/Julia-1):
`state + question + 2–20 options → one probability per option` (`choice`, `score`, `noul`).

What it reimplements, from the repo's Python (`julia/model.py`, `data.py`, `typed.py`):

* `safetensors` loader (mmap; F32 / F16 / BF16), JSON parser
* Gemma/mmBERT BPE tokenizer driven by the real `tokenizer.json` (added tokens, `Replace`
  normalizer, byte fallback, merge ranks)
* the exact `sequence()` encoding: `<bos> "type question: …" <eos> [<mask> option]… <eos> state <eos>`
* ModernBERT encoder: 22 layers, RoPE, alternating global / ±64 sliding-window attention, GeGLU
* decision head: type embedding → 2 pre-norm transformer layers (ReLU FFN) → scorer MLP
* typed API (`answers → choice / score / noul / probabilities / max_probability`) and the legacy list API



## Dependencies:

```sh
julia-1.cpp
  |-- config.json           # copy from huggingface.co/SupersonicLabs/Julia-1/encoder folder
  |-- example.json
  |-- json.c
  |-- json.h
  |-- julia.c
  |-- julia.h
  |-- tokenizer.c
  |-- tokenizer.h
  |-- main.c
  |-- julia_config.json     # copy from huggingface.co/SupersonicLabs/Julia-1
  |-- Makefile
  |-- README.md
  |-- model.safetensors     # copy from huggingface.co/SupersonicLabs/Julia-1 
  |-- tokenizer_config.json # copy from huggingface.co/SupersonicLabs/Julia-1/tokenizer folder
  |-- tokenizer.json        # copy from huggingface.co/SupersonicLabs/Julia-1/tokenizer folder
```



## Build:

```sh
make OMP=1    # or `make` for single-threaded; drop -march=native in the Makefile for a portable binary
./julia-1 . example.json
./julia-1 . --serve < requests.jsonl      # model stays loaded, one JSON per line
```



## Run:

```sh
./julia-1 . example.json
./julia-1 . --serve < requests_in_json      # model stays loaded, one JSON per line
```

Other run options: `--legacy`, `--max-length N` (8192), `--head-length N` (512), `--no-strict`, `--threads N`,
`--debug` (adds ids / marker positions / raw logits), `--info`, `--list-tensors`, `--tokens TEXT`.
