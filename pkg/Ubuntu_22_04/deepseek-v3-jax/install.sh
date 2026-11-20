git clone https://github.com/Vaibhaav-Tiwari/deepseek-v3-jax.git && cd deepseek-v3-jax
uv venv -p 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

JAX_PLATFORMS=cpu .venv/bin/pytest tests.py -q    # 39 passed
.venv/bin/python smoke_metal.py                   # optional: what jax-metal supports on your Mac

mkdir -p data
for f in train.bin val.bin tokenizer.json; do curl -L -o data/$f https://github.com/Vaibhaav-Tiwari/deepseek-v3-jax/releases/download/v1.0/$f; done
shasum -a 256 data/*.bin data/tokenizer.json    # should match the hashes above

.venv/bin/python train.py                                         # the main run: 12k steps, ~2.6 h
.venv/bin/python train.py --steps 2000 --out runs/bias_on         # balancing ablation, with the bias
.venv/bin/python train.py --steps 2000 --out runs/bias_off --no-bias   # ...and without it

.venv/bin/python sample.py --prompt "The fog came in off the" --num 3

.venv/bin/python plots.py                    # needs runs/main, runs/bias_on, runs/bias_off
JAX_PLATFORMS=cpu .venv/bin/python charts.py # the rest of the write-up's charts
