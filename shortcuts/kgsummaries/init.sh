# cat s2link | while read line; do title="$(echo $line | sed 's/\(.*\) -> \(.*\)/\1/g')"; link="$(echo $line | sed 's/\(.*\) -> \(.*\)/\2/g')"; md5=$(echo $title | md5sum | awk '{print $1}'); done | tee /tmp/s2md5
# [ "$(md5sum $(ltr | tail -n1 | tail -n1 | awk '{print $NF}') | awk '{print $1}')" = "$(md5sum $(ltr | tail -n2 | head -n1 | awk '{print $NF}') | awk '{print $1}')" ]
# [ "$(ps aux G short G kgsummaries.3)" != "" ]
