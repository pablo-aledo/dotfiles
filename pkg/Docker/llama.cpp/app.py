from llama_cpp import Llama

__VERBOSE = True
__N_CTX = 1024
__MAX_TOKENS = 4096 - __N_CTX

model = Llama(
    model_path="mistral-7b-openorca.Q5_K_M.gguf",
    n_gpu_layers=35,
    n_ctx=__N_CTX,
    n_threads=1,
    verbose=__VERBOSE,
)

template = """
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

prompt = "How did the Universe start?"

llm_res = model(
    template.replace(
        "{system}",
        "You are answering the questions of someone with a Physics degree. Please be very technical.",
    ).replace(
        "{prompt}",
        prompt,
    ),
    max_tokens=__MAX_TOKENS,
)

print("")
print("User:", prompt)
print("Mistral-7B:", llm_res["choices"][0]["text"])
