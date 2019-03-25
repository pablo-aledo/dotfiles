go get github.com/atsaki/termeter/cmd/termeter
#  seq 100 | awk 'BEGIN{OFS="\t"; print "x","sin(x)","cos(x)"}{x=$1/10; print x,sin(x),cos(x)}' | termeter
# seq 300 | awk 'BEGIN{OFS="\t"; print "x","sin(x)","cos(x)"}{x=$1/10; print x,sin(x),cos(x); system("sleep 0.1")}' | termeter
# (echo "line counter cdf"; seq 1 1000 | awk '{x=int(6*rand())+1; print x,x,x}') | termeter -d " " -t lcd -S numerical
