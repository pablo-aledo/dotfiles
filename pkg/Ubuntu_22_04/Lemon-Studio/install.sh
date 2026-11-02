# Debian / Ubuntu
sudo apt install g++ libclang-dev libasound2-dev libjack-jackd2-dev libwayland-dev pkg-config
# Arch (pipewire-jack or jack2)
# sudo pacman -S gcc clang alsa-lib pipewire-jack wayland pkgconf
# Fedora
# sudo dnf install gcc-c++ clang-devel alsa-lib-devel pipewire-jack-audio-connection-kit-devel wayland-devel pkgconf-pkg-config

cargo run -p os-app --release

#cargo install --path crates/os-app
#install -Dm644 crates/os-app/assets/lemonstudio.png ~/.local/share/icons/hicolor/256x256/apps/lemonstudio.png
#install -Dm644 crates/os-app/assets/lemonstudio.desktop ~/.local/share/applications/lemonstudio.desktop
