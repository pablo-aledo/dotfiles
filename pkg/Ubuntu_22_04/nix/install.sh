sh <(curl -L https://nixos.org/nix/install) --daemon
mkdir -p ~/.config/nix/
echo 'experimental-features = nix-command flakes' > ~/.config/nix/nix.conf
echo 'export PATH=/nix/var/nix/profiles/default/bin:$PATH' >> ~/.paths
