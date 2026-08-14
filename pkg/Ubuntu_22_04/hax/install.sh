git clone https://github.com/OleksandrChekhovskyi/hax.git
cd hax
scripts/install_deps.sh   # Debian/Ubuntu, Fedora, Arch, openSUSE, Alpine, macOS
make                      # the binary is now at ./build/hax
make install              # optional; may prompt for sudo
