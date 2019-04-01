FROM fedora:29

run dnf -y update && \
    dnf -y install postgresql-contrib

#copy pgdump.tgz        /
copy restore_script.sh /
run chmod +x /restore_script.sh

entrypoint "/restore_script.sh"

