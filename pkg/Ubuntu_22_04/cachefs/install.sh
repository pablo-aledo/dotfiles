sudo apt install fuse
pip install fusepy
mkdir -p ~/.cache/redfs ~/mnt/cachefs
python3 cachefs.py /mnt2/redfs ~/.cache/redfs ~/mnt/cachefs
