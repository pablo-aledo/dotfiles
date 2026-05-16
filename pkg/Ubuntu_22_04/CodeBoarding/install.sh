sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev

sudo apt install -y pipx
pipx ensurepath
source ~/.bashrc

pipx install codeboarding --python python3.12

codeboarding-setup

vim ~/.codeboarding/config.toml

# codeboarding full --local ~/partituras
# codeboarding full --local ~/partituras --depth-level 2
# codeboarding incremental --local ~/partituras

# local
#    curl -LsSf https://astral.sh/uv/install.sh | sh
#    source ~/.bashrc
#
#    unzip main.zip
#    cd CodeBoarding-main
#
#    uv sync --frozen
#
#    source .venv/bin/activate
#    python install.py
#
#    python main.py full --local ~/partituras
