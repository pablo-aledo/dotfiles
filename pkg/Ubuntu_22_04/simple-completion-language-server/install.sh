cargo install --git https://github.com/estin/simple-completion-language-server.git

cat < EOF > ~/.config/helix/languages.toml
# introduce new language server
[language-server.scls]
command = "simple-completion-language-server"

[language-server.scls.config]
max_completion_items = 20            # set max completion results len for each group: words, snippets, unicode-input
snippets_first = true                # completions will return before snippets by default
snippets_inline_by_word_tail = false # suggest snippets by WORD tail, for example text 'xsq|' become 'x^2|' when snippet 'sq' has body '^2'
feature_words = true                 # enable completion by word
feature_snippets = true              # enable snippets
feature_unicode_input = true         # enable "unicode input"
feature_paths = true                 # enable path completion
feature_citations = false            # enable citation completion (only on 'citation' feature enabled)


# write logs to /tmp/completion.log
[language-server.scls.environment]
RUST_LOG = "info,simple-completion-language-server=info"
LOG_FILE = "/tmp/completion.log"

# append language server to existed languages
[[language]]
name = "rust"
language-servers = [ "scls", "rust-analyzer" ]

[[language]]
name = "git-commit"
language-servers = [ "scls" ]

# etc..

# introduce a new language to enable completion on any doc by forcing set language with :set-language stub
[[language]]
name = "stub"
scope = "text.stub"
file-types = []
shebangs = []
roots = []
auto-format = false
language-servers = [ "scls" ]
EOF
