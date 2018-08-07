set prompt \033[01;31m\n\n#######################################> \033[0m

shell mkfifo /tmp/colorPipe_list
shell mkfifo /tmp/colorPipe_backtrace
shell mkfifo /tmp/colorPipe_next
shell mkfifo /tmp/colorPipe_step

define hook-list
echo \n
shell cat /tmp/colorPipe_list | highlight --syntax=cpp -O ansi &
set logging redirect on
set logging on /tmp/colorPipe_list
end

define hookpost-list
set logging off
set logging redirect off
shell sleep 0.1s
end

define hook-step
echo \n
shell cat /tmp/colorPipe_step | highlight --syntax=cpp -O ansi &
set logging redirect on
set logging on /tmp/colorPipe_step
end

define hookpost-step
set logging off
set logging redirect off
shell sleep 0.1s
end

define hook-next
echo \n
shell cat /tmp/colorPipe_next | highlight --syntax=cpp -O ansi &
set logging redirect on
set logging on /tmp/colorPipe_next
end

define hookpost-next
set logging off
set logging redirect off
shell sleep 0.1s
end

define hook-backtrace
echo \n
shell cat /tmp/colorPipe_backtrace | sed -e "s/\(.*\)\(\<.*\>.\<.*\>:\<.*\>\)$/\1\o033[1;31m\2\o033[0m/g" -e "s/\([^:]*\)\<\(.*\)\((.*\)/\1\o033[0;32m\2\o033[0m\3/g" &
set logging redirect on
set logging on /tmp/colorPipe_backtrace
end

define hookpost-backtrace
set logging off
set logging redirect off
shell sleep 0.1s
end

define hook-quit
shell rm /tmp/colorPipe_list
shell rm /tmp/colorPipe_backtrace
shell rm /tmp/colorPipe_next
shell rm /tmp/colorPipe_step
end
