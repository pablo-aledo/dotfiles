git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
printf 'y\ny\ny\n' | ~/.fzf/install
sed -i 's/--scheme=history/--tiebreak=index/g' ~/.fzf/shell/key-bindings.zsh
sudo cp rfv /usr/bin/rfv; sudo chmod +x /usr/bin/rfv
