sudo apt-get install surfraw surfraw-extra
mkdir .config/surfraw
echo 'SURFRAW_graphical_browser=/usr/bin/chromium' > ~/.config/surfraw/conf
echo 'SURFRAW_text_browser=/usr/bin/w3m' >> ~/.config/surfraw/conf
echo 'SURFRAW_graphical=no' >> ~/.config/surfraw/conf
