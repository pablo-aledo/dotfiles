cd

pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118
git clone https://github.com/turboderp/exllama
cd exllama

pip install -r requirements.txt

python test_benchmark_inference.py -d "<path_to_model_files>" -p -ppl
python example_chatbot.py -d "<path_to_model_files>" -un "Jeff" -p prompt_chatbort.txt
