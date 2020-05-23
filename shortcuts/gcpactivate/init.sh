[ $# -ge 1 ] && gcloud_user=$1
[ $# -ge 1 ] || gcloud_user=""

[ -e ~/Dotfiles ] || pf_mount Dotfiles

if [ "$gcloud_user" = "" ]
then
    base_user=$(echo $USER | sed 's/[0-9]*$//g')
    num_user=$(echo $USER | sed 's/^[^0-9]*//g')
    gcloud_user=$(echo $base_user | sed 's/_/./g' )$(( $num_user + 1 ))
fi

sed -i "s/gcloud_user=.*/gcloud_user=$gcloud_user/g" ~/Dotfiles/remote/setenv

cd ~/.dotfiles/shortcuts/gcpactivate/
