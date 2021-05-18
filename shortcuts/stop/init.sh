sed -i.bak '/bindsym.*\/shortcuts\/.*/d' ~/.i3/config
tmux kill-session -t novnc
tmux kill-session -t vnc
