[ -e /tmp/vimrc ] || cp ~/.vimrc /tmp/vimrc
rm -rf ~/.vim ~/.vimrc

git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim

cat << EOF >> ~/.vimrc
set nocompatible              " be iMproved, required
filetype off                  " required

" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'

" The following are examples of different formats supported.
" Keep Plugin commands between vundle#begin/end.
" plugin on GitHub repo

Plugin 'vim-airline/vim-airline'
Plugin 'vim-scripts/Align'
Plugin 'vim-scripts/AnsiEsc.vim'
Plugin 'Townk/vim-autoclose'
Plugin 'vim-scripts/a.vim'
Plugin 'Rip-Rip/clang_complete'
Plugin 'vim-scripts/Conque-Shell'
Plugin 'kien/ctrlp.vim'
Plugin 'Twinside/vim-cuteErrorMarker'
Plugin 'c.vim'
Plugin 'fadein/vim-FIGlet'
Plugin 'tpope/vim-fugitive'
Plugin 'sjl/gundo.vim'
Plugin 'tmhedberg/matchit'
Plugin 'severin-lemaignan/vim-minimap'
Plugin 'scrooloose/nerdcommenter'
Plugin 'scrooloose/nerdtree'
Plugin 'vim-scripts/OmniCppComplete'
Plugin 'kien/rainbow_parentheses.vim'
Plugin 'LucHermitte/vim-refactor'
Plugin 'derekwyatt/vim-scala'
Plugin 'vim-scripts/SearchComplete'
Plugin 'vim-scripts/self.vim'
Plugin 'garbas/vim-snipmate'
Plugin 'tpope/vim-surround'
Plugin 'vim-syntastic/syntastic'
Plugin 'majutsushi/tagbar'
Plugin 'taglist.vim'
Plugin 'edkolev/tmuxline.vim'
Plugin 'jgdavey/tslime.vim'
Plugin 'vim-scripts/AutoComplPop'
Plugin 'gyim/vim-boxdraw'
Plugin 'ryanoasis/vim-devicons'
Plugin 'tpope/vim-dispatch'
Plugin 'airblade/vim-gitgutter'
Plugin 'vim-latex/vim-latex'
Plugin 'triglav/vim-visual-increment'

" Plugin 'tpope/vim-fugitive'
" plugin from http://vim-scripts.org/vim/scripts.html
" Plugin 'L9'
" Git plugin not hosted on GitHub
" Plugin 'git://git.wincent.com/command-t.git'
" git repos on your local machine (i.e. when working on your own plugin)
" Plugin 'file:///home/gmarik/path/to/plugin'
" The sparkup vim script is in a subdirectory of this repo called vim.
" Pass the path to set the runtimepath properly.
" Plugin 'rstacruz/sparkup', {'rtp': 'vim/'}
" Install L9 and avoid a Naming conflict if you've already installed a
" different version somewhere else.
" Plugin 'ascenator/L9', {'name': 'newL9'}

" All of your Plugins must be added before the following line
call vundle#end()            " required
filetype plugin indent on    " required
" To ignore plugin indent changes, instead use:
"filetype plugin on
"
" Brief help
" :PluginList       - lists configured plugins
" :PluginInstall    - installs plugins; append `!` to update or just :PluginUpdate
" :PluginSearch foo - searches for foo; append `!` to refresh local cache
" :PluginClean      - confirms removal of unused plugins; append `!` to auto-approve removal
"
" see :h vundle for more details or wiki for FAQ
" Put your non-Plugin stuff after this line
EOF
vim +PluginInstall +qall

cat /tmp/vimrc >> ~/.vimrc
