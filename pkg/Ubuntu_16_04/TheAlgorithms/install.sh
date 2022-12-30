# cat repositories G '/TheAlgorithms' G 'name codeRepository' | cut -d'"' -f6 | cut -d/ -f3

for lang in *
do
    [ $lang = install.sh ] && continue
    mkdir $lang
    cd $lang
    zipurl="https://github.com/TheAlgorithms/$lang/archive/refs/heads/master.zip"
    wget $zipurl
    cd ..
done
