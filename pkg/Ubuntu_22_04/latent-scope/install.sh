python -m venv venv
source venv/bin/activate
pip install latentscope
ls-init ~/local-scope-data --openai_key=$OPENAI_API_KEY # --mistral_key=YYY # optional api keys to enable API models
ls-serve
