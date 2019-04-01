cd /tmp
tar -xvzf /pgdump.tgz

cd tmp

for a in *
do
    pg_restore -d postgres://postgres:postgres@postgres:5432/$a $a
done

tail -f /dev/null
