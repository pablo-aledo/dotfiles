for lang in *
do
    [ $lang = install.sh ] && continue
    mkdir $lang
    cd $lang
    #googler "raytracer in one weekend $lang site: github.com" --json | grep '"url":' | cut -d'"' -f4 > urls
    cat urls | head -n99 | while read url
    do
        zipurl="$url/archive/refs/heads/main.zip"
        name="$(echo $url | cut -d/ -f4).$(echo $url | cut -d/ -f5).zip"
        [ -e $name ] || wget $zipurl -O $name
    done
    cd ..
done
