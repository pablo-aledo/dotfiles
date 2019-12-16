wget https://raw.githubusercontent.com/rupa/z/master/z.sh -O ~/.z.sh 
echo 'source ~/.z.sh' >> ~/.shell 
[ -d /media/DATA/z  ] && ( rm -fr ~/.z ; ln -s /media/DATA/z/.z ~/.z )
[ -f /media/DATA/.z ] && ( rm -fr ~/.z ; ln -s /media/DATA/.z   ~/.z )

echo 'if [ "$_Z_NO_RESOLVE_SYMLINKS" ]; then' >> ~/.shell
echo '    _z_precmd() {' >> ~/.shell
echo '        (_z --add "${PWD:a}" &)' >> ~/.shell
echo '		: $RANDOM' >> ~/.shell
echo '    }' >> ~/.shell
echo 'else' >> ~/.shell
echo '    _z_precmd() {' >> ~/.shell
echo '        (_z --add "${PWD:A}" &)' >> ~/.shell
echo '		: $RANDOM' >> ~/.shell
echo '    }' >> ~/.shell
echo 'fi' >> ~/.shell
