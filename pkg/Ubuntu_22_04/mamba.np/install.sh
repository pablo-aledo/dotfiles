pip install torch
cd
git clone https://github.com/idoh/mamba.np.git
cd mamba.np
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python mamba.py "I have a dream that"
