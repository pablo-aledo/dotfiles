```
docker build -t rectalogic/neural-style:latest .

git clone https://github.com/jcjohnson/neural-style.git
cd neural-style
bash models/download_models.sh

docker run -i -v $PWD/neural-style:/neural-style -t rectalogic/neural-style:latest -gpu -1 -style_image examples/inputs/starry_night.jpg -content_image examples/inputs/hoovertowernight.jpg
```
