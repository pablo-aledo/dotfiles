cd
mkdir cupido
cd cupido
wget https://github.com/williamfzc/cupido/releases/download/v0.3.3/cupido-x86_64-unknown-linux-musl.zip
unzip cupido-x86_64-unknown-linux-musl.zip
./cupido up --repo-path ~/workspace/github/axios
#curl http://127.0.0.1:9410/size
#curl http://127.0.0.1:9410/file/-/issues?file=file%2F
#curl http://127.0.0.1:9410/file/-/commits?file=file%2F
#curl http://127.0.0.1:9410/issue/-/files?issue=%23issue
#curl http://127.0.0.1:9410/issue/-/commits?issue=%23issue
#curl http://127.0.0.1:9410/commit/-/files?commit=file%2F
#curl http://127.0.0.1:9410/commit/-/issues?commit=file%2F
#curl http://127.0.0.1:9410/issue/list
#curl http://127.0.0.1:9410/author/-/commits?author=author
#curl http://127.0.0.1:9410/author/list
#curl http://127.0.0.1:9410/commit/-/authors?commit=commit
