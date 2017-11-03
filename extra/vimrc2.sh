for a in 00_basic 01_plugins 02_cscope 03_ctags 04_ctrlp 05_gui 06_autocmd 07_maps 07_movement 07_tmux 08_header 09_airline 10_duplilines 11_highlight 12_search
do
	wget https://raw.githubusercontent.com/pablo-aledo/dotfiles/master/source/.vimrc/$a -O - >> /tmp/vimrc_pga ;
done

echo "alias vim='vim +\":so /tmp/vimrc_pga\"'"
