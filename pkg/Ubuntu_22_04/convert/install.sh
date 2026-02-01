pkg install imagemagick

sudo sed -i 's/^ *<policy \(.*\)$/  <!-- <policy\1 -->/g' /etc/ImageMagick-6/policy.xml
