wget https://raw.githubusercontent.com/rupa/z/master/z.sh -O ~/.z.sh 
echo 'source ~/.z.sh' >> ~/.shell 
[ -d /media/DATA/z  ] && ( rm -fr ~/.z ; ln -s /media/DATA/z/.z ~/.z )
[ -f /media/DATA/.z ] && ( rm -fr ~/.z ; ln -s /media/DATA/.z   ~/.z )

# echo << EOF >> ~/.shell
# if [ "$_Z_NO_RESOLVE_SYMLINKS" ]; then
#     _z_precmd() {
#         (_z --add "${PWD:a}" &)
# 		: $RANDOM
#     }
# else
#     _z_precmd() {
#         (_z --add "${PWD:A}" &)
# 		: $RANDOM
#     }
# fi
# EOF
