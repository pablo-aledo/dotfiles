git_branches="git branch --all --color \
  --format=$'%(HEAD) %(color:yellow)%(refname:short)\t%(color:green)%(committerdate:short)\t%(color:blue)%(subject)' \
  | column --table --separator=$'\t'" \
&& eval "$git_branches" \
| fzf \
  --ansi \
  --reverse \
  --no-sort \
  --preview-label='[ Commits ]' \
  --preview='git log $(echo {} \
    | sed "s/^[* ]*//" \
    | awk "{print \$1}") \
    --graph --color \
    --format="%C(white)%h - %C(green)%cs - %C(blue)%s%C(red)%d"' \
  --bind='alt-c:execute(
    git checkout $(echo {} \
    | sed "s/^[* ]*//" \
    | awk "{print \$1}")
    )' \
  --bind="alt-c:+reload($git_branches)" \
  --bind='alt-m:execute(git merge $(echo {} \
    | sed "s/^[* ]*//" \
    | awk "{print \$1}")
    )+abort' \
  --bind='alt-r:execute(git rebase $(echo {} \
    | sed "s/^[* ]*//" \
    | awk "{print \$1}")
    )+abort' \
  --bind='enter:execute(
    branch=$(echo {} \
    | sed "s/^[* ]*//" \
    | awk "{print \$1}") \
    && sh -c "git diff --color $branch | less -R"
    )' \
  --header-first \
  --header '
  > The branch marked with a star * is the current branch
  > ALT-C to checkout the branch
  > ALT-M to merge with current branch | ALT-R to rebase with current branch
  > ENTER to open the diff with less
  '
