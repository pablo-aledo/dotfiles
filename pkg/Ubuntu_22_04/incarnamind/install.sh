cd
git clone https://github.com/junruxiong/IncarnaMind
cd IncarnaMind
conda create -n IncarnaMind python=3.10
conda activate IncarnaMind
pip install -r requirements.txt
python docs2db.py
python main.py
