cd /usr/share
wget https://github.com/asciidocfx/AsciidocFX/releases/download/v1.5.9/AsciidocFX_Linux.tar.gz -O - | sudo tar -xvz
echo '#!/bin/bash'               | sudo tee    /usr/bin/asciidocfx
echo 'cd /usr/share/AsciidocFX/' | sudo tee -a /usr/bin/asciidocfx
echo 'sudo ./AsciidocFX'         | sudo tee -a /usr/bin/asciidocfx
sudo chmod +x /usr/bin/asciidocfx
