echo '#!/usr/bin/zsh'  | sudo tee /usr/bin/zsh_source_run
echo 'source ~/.shell' | sudo tee -a /usr/bin/zsh_source_run
echo 'source ~/.paths' | sudo tee -a /usr/bin/zsh_source_run
echo '$*'              | sudo tee -a /usr/bin/zsh_source_run

sudo chmod +x /usr/bin/zsh_source_run
