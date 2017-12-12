sudo apt-get install -y mutt

mkdir -p ~/.mutt/cache/headers
mkdir ~/.mutt/cache/bodies
touch ~/.mutt/certificates
touch ~/.mutt/muttrc

cat << EOF > ~/.mutt/muttrc
set ssl_starttls=yes
set ssl_force_tls=yes
set imap_user = 'pablo.aledo@gmail.com'
set from='pablo.aledo@gmail.com'
set realname='Pablo Gonzalez de Aledo'
set folder = imaps://imap.gmail.com/
set spoolfile = imaps://imap.gmail.com/INBOX
set postponed="imaps://imap.gmail.com/[Gmail]/Drafts"
set header_cache = "~/.mutt/cache/headers"
set message_cachedir = "~/.mutt/cache/bodies"
set certificate_file = "~/.mutt/certificates"
set smtp_url = 'smtps://pablo.aledo@smtp.gmail.com:465/'
set move = no
set imap_keepalive = 900
set imap_pass = ''
set smtp_pass = ''
EOF

[ -e ~/Dotfiles/mutt/muttrc ]      && \cp ~/Dotfiles/mutt/muttrc ~/.mutt/muttrc
[ -e /media/DATA/Personal/muttrc ] && \cp /media/DATA/Personal/muttrc ~/.mutt/muttrc
[ -e $REPOSITORY_FOLDER/mutt/`distr`/mutt/muttrc ] && \cp $REPOSITORY_FOLDER/mutt/`distr`/mutt/muttrc ~/.mutt/muttrc

echo "https://security.google.com/settings/security/apppasswords"
