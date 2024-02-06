[ $# -eq 0 ] && link="https://learn.deeplearning.ai/langchain"

[ $# -eq 1 ] && link=$1

echo $link > /tmp/link
echo 0 > /tmp/filenr
