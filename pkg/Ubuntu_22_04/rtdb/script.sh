while true
do
    echo -n "."
    # get logs
    source innerloop.sh
    sleep 1
    [ "$(cat graph.im | md5sum)" != "$(cat graph.final | md5sum)" ] && \cp graph.im graph.final
done

