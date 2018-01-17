echo 'export PROMPT_COMMAND="pwd > /tmp/whereami"' >> ~/.bashrc
echo 'precmd(){ pwd > /tmp/whereami }' >> ~/.shell

echo '#!/bin/zsh' > /home/mint/.i3/i3_shell.sh
echo 'gnome-terminal --working-directory="$(cat /tmp/whereami)"' >> /home/mint/.i3/i3_shell.sh
chmod +x /home/mint/.i3/i3_shell.sh

sed -i 's@bindsym $mod+Return exec i3-sensible-terminal@bindsym $mod+Return exec $HOME/.i3/i3_shell.sh@g' ~/.i3/config

i3-msg reload
