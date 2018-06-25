# Docker container for antlr

"ANTLR (ANother Tool for Language Recognition) is a powerful parser generator for reading, processing, executing, or translating structured text or binary files. It's widely used to build languages, tools, and frameworks. From a grammar, ANTLR generates a parser that can build and walk parse trees."

www.antlr.org

## Setup

Use the docker-antlr helper script to fire up the container conveniently. docker-antlr accepts the very same command line options as antlr itself. it bind-mounts all the necessary directories (directories of your input and output files) and calls antlr within the container. You can get docker-antlr form the git repository or copy it directly from the container's root directory.

```
docker create --name antlr petervaczi/antlr
sudo docker cp antlr:docker-antlr /usr/local/bin/
docker rm antlr
```

### Usage

Use the docker-antlr script exactly as you would use antlr itself.

```
docker-antlr <antlr-options>
```
