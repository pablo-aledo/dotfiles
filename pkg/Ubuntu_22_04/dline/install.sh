pkg install at
pkg install jq
mkd ~/dline
wget https://github.com/jazz-it/dline/archive/refs/heads/main.zip
unzip main.zip
mv dline-main/* .
rm -fr dline-main main.zip
