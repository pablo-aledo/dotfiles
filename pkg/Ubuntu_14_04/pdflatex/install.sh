sudo apt-get install -y texlive `apt-cache search texlive | grep -E 'texlive-.*' | cut -d" " -f1 | grep -v doc | grep -v lang | grep -v full | grep -v games` pgf latex-beamer latexmk
