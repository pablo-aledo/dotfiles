git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
printf 'y\ny\ny\n' | ~/.fzf/install
source ~/.fzf.bash

wget https://rawgit.com/kubermatic/fubectl/master/fubectl.source -O ~/.fubectl
echo 'source ~/.fubectl' >> ~/.shell
source ~/.fubectl
