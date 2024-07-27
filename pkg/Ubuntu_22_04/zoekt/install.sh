go install github.com/sourcegraph/zoekt@latest
go install github.com/sourcegraph/zoekt/cmd/zoekt-index@latest
go install github.com/sourcegraph/zoekt/cmd/zoekt-git-index@latest
go install github.com/sourcegraph/zoekt/cmd/zoekt-{repo-index,mirror-gitiles}@latest
go install github.com/sourcegraph/zoekt/cmd/zoekt-webserver@latest
go install github.com/sourcegraph/zoekt/cmd/zoekt@latest
go install github.com/sourcegraph/zoekt/cmd/zoekt-indexserver@latest

zoekt-index .
# zoekt-git-index -branches master -prefix origin/ .
# zoekt-mirror-gitiles -dest ~/repos/ https://gfiber.googlesource.com
# zoekt-repo-index \
#     -name gfiber \
#     -base_url https://gfiber.googlesource.com/ \
#     -manifest_repo ~/repos/gfiber.googlesource.com/manifests.git \
#     -repo_cache ~/repos \
#     -manifest_rev_prefix=refs/heads/ --rev_prefix= \
#     master:default_unrestricted.xml

cp *.zoekt ~/.zoekt

zoekt-webserver -listen :6070

# zoekt 'ngram f:READ'
#
# curl --get \
#     --url "http://localhost:6070/search" \
#     --data-urlencode "q=ngram f:READ" \
#     --data-urlencode "num=50" \
#     --data-urlencode "format=json"
#
# zoekt-indexserver -mirror_config config.json
