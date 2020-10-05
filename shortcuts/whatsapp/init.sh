echo $1 > /tmp/whatsapp_contact
shift

for a in $*
do
    echo $a | sed "s|^|$PWD/|g"
done > /tmp/whatsapp_files
