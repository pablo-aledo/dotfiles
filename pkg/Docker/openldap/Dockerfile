FROM alpine

RUN apk add --no-cache -q openldap openldap-back-bdb openldap-clients openldap-overlay-all

COPY ./run.sh /usr/bin/run.sh
RUN  chmod +x /usr/bin/run.sh
COPY ./extdb.template /etc/extdb.template

ENTRYPOINT run.sh
