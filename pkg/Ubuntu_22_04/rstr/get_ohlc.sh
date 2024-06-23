export DISPLAY=:1

x=0
xdotool mousemove $x 100
for x_scan in $(seq 0 1280)
do
    xdotool mousemove_relative 1 0
    rm -fr /tmp/screenshot.png
    scrot -z /tmp/screenshot.png
    [ $(convert /tmp/screenshot.png -format "%[hex:u.p{5,737}]\n" info:) = 363A45 ] && break
done

x_date_1=0
x_date_2=114
y_date_1=737
y_date_2=754
for x_scan in $(seq 0 1280)
do
    xdotool mousemove_relative 1 0
    rm -fr /tmp/screenshot.png
    scrot -z /tmp/screenshot.png

    convert -crop 1000x1+0+$y_date_1 /tmp/screenshot.png /tmp/horizontal_bar.png

    for x_date_1 in $(seq $(( $x_date_1 - 10 )) $(( $x_date_1 + 10 )))
    do
        [ $x_date_1 -lt 0 ] && continue
        [ $(convert /tmp/horizontal_bar.png -format "%[hex:u.p{$x_date_1,1}]\n" info:) = 363A45 ] && break
    done
    for x_date_2 in $(seq $(( $x_date_2 + 10 )) -1 $(( $x_date_1 - 10 )))
    do
        [ $(convert /tmp/horizontal_bar.png -format "%[hex:u.p{$x_date_2,1}]\n" info:) = 363A45 ] && break
    done

    convert -channel RGB -negate -crop $(($x_date_2 - $x_date_1 - 2))x$(($y_date_2 - $y_date_1))+$(( $x_date_1  + 1))+$y_date_1 /tmp/screenshot.png /tmp/date.png
    date=$((tesseract -l eng /tmp/date.png - | head -n1) 2>/dev/null)
    date=$(echo $date | sed 's/\([0-9]*\) *\.*\([a-zA-Z]*\)[^0-9]*\([0-9]*\) \([0-9]*:[0-9]*\).*/\1 \2 \3 \4/g')
    date=$(date --date="$(echo $date | awk '{print $2" "$1" "20$3" "$4}')" +"%s")

    x_ohlc_1=40
    y_ohlc_1=8
    x_ohlc_2=360
    y_ohlc_2=32
    convert -channel RGB -negate -crop $(($x_ohlc_2 - $x_ohlc_1 - 2))x$(($y_ohlc_2 - $y_ohlc_1))+$(( $x_ohlc_1  + 1))+$y_ohlc_1 /tmp/screenshot.png /tmp/ohlc.png
    ohlc=$((tesseract -l eng /tmp/ohlc.png - | head -n1 | sed -e 's/^©/O/g' -e 's/^0/O/g' -e 's/O/, /g' -e 's/ H/, /g' -e 's/ L/, /g' -e 's/ €/, /g' -e 's/ ©/, /g' -e 's/ C/, /g') 2>/dev/null)
    ohlc=$(echo $ohlc | sed 's/\([0-9]\) \([0-9]\)/\1, \2/g')
    ohlc=$(echo $ohlc | sed -e 's/, \([^ ]*\), \([^ ]*\), \([^ ]*\), \([^ ]*\) .*/\1, \2, \3, \4/g')
    ohlc=$(echo $ohlc | sed -e 's/\([^ ]*\), \([^ ]*\), \([^ ]*\), \([^ ]*\) .*/\1, \2, \3, \4/g')
    ohlc=$(echo $ohlc | sed -e 's/,$//g')

    echo "$date, $ohlc" | tee -a $1

done

