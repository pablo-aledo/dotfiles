[ $# -ge 1 ] && cloud_user=$1
[ $# -ge 1 ] || cloud_user=""

[ -e ~/Dotfiles ] || pf_mount Dotfiles

if [ "$cloud_user" = "" ]
then
    base_user=$(echo $USER | sed 's/[0-9]*$//g')
    num_user=$(echo $USER | sed 's/^[^0-9]*//g')
    gcloud_user=$(echo $base_user | sed 's/_/./g' )$(( $num_user + 1 ))
fi

sed -i "s/cloud_user=.*/cloud_user=$gcloud_user/g" ~/Dotfiles/remote/setenv
