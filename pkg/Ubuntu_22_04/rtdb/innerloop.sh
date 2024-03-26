rmq(){
    sed -e 's/"//g'
}

pga_vis_clear(){
    group=$(echo $line | jq '.group' | rmq)
    sed -i "/$group\./d" graph.im
}

pga_vis_rm(){
    group=$(echo $line | jq '.group' | rmq)
    key=$(echo $line | jq '.key' | rmq)
    sed -i "/$group\.\"$key\"/d" graph.im
}

pga_vis_update(){
    group=$(echo $line | jq '.group' | rmq)
    key=$(echo $line | jq '.key' | rmq)

    sed -i "/$group\.\"$key\"/d" graph.im

    echo "${group}.\"${key}\"" >> graph.im
    echo -n "${group}.\"${key}\": \"${key} " >> graph.im
    for a in $*
    do
        value=$(echo $line | jq ".$a" | rmq)
        echo -n "\\\\n$a:$value " >> graph.im
    done
    echo '"' >> graph.im
}

pga_vis_add(){
    group=$(echo $line | jq '.group' | rmq)
    key=$(echo $line | jq '.key' | rmq)
    echo "${group}.\"${key}\"" >> graph.im
    echo -n "${group}.\"${key}\": \"${key} " >> graph.im
    for a in $*
    do
        echo -n "\"$a\":"
        value=$(echo $line | jq ".$a" | rmq)
        echo -n "\\\\n$a:$value " >> graph.im
    done
    echo '"' >> graph.im
}

pga_vis_link(){
    group1=$(echo $line | jq '.group1' | rmq)
    key1=$(echo $line | jq '.key1' | rmq)
    group2=$(echo $line | jq '.group2' | rmq)
    key2=$(echo $line | jq '.key2' | rmq)
    echo "${group1}.\"${key1}\" -> ${group2}.\"${key2}\" " >> graph.im
}

echo -n '' > graph.im
cat /tmp/log.log* | grep pga_vis | while read line
do
    [ "$(echo $line | grep pga_vis | grep clear)" != "" ] && { pga_vis_clear; continue }
    [ "$(echo $line | grep pga_vis | grep .operation...rm )" != "" ] && { pga_vis_rm; continue }
    [ "$(echo $line | grep pga_vis | grep group)" != "" ] && { pga_vis_add field; continue }
done

# extra groups
