[ -e /usr/bin/gvim ] || echo 'sudo apt-get remove -y vim-gnome' > ../../../uninstall/gvim

sudo apt-get install -y vim-gnome exuberant-ctags
