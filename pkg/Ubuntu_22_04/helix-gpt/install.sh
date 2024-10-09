wget https://github.com/leona/helix-gpt/releases/download/0.34/helix-gpt-0.34-x86_64-linux.tar.gz -O /tmp/helix-gpt.tar.gz \
&& tar -zxvf /tmp/helix-gpt.tar.gz \
&& sudo mv helix-gpt-0.34-x86_64-linux /usr/bin/helix-gpt \
&& sudo chmod +x /usr/bin/helix-gpt

cat <<EOF >> ~/.helix/languages.toml
[language-server.gpt]
command = "helix-gpt"

[language-server.ts]
command = "typescript-language-server"
args = ["--stdio"]
language-id = "javascript"

[[language]]
name = "typescript"
language-servers = [
    "ts",
    "gpt"
]
EOF

# echo 'export OPENAI_API_KEY="..."' >> ~/.paths
echo 'export HANDLER=openai' >> ~/.paths

