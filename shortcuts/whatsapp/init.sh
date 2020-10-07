echo $1 > /tmp/whatsapp_contact
shift

for a in $*
do
    echo $a | sed "s|^|$PWD/|g"
done > /tmp/whatsapp_files

export DISPLAY=:1
source ~/.dotfiles/source/.shell/adapt
xdotool key 1; shcw 1
xdotool key 2; shcw 2
xdotool key 3; shcw 3
xdotool key 4; shcw 4
xdotool key 0
