pkg install ghc
pkg install cabal-install

cd /tmp
git clone https://github.com/fmthoma/ascii-art-to-unicode.git
cd ascii-art-to-unicode

cabal update
cabal install

echo export PATH=$HOME/.cabal/bin:$PATH >> ~/.paths
export PATH=$HOME/.cabal/bin:$PATH

