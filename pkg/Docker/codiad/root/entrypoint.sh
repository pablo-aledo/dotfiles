#!/bin/sh

# Install Codiad if not already installed.
if [ ! -d '/code/.git' ]; then
    cp -r /default-code/. /code
    chmod go+w \
        /code/config.php \
        /code/workspace \
        /code/plugins \
        /code/themes \
        /code/data
fi

# Set user:group ID.
CODIAD_UID=${CODIAD_UID:-2743}
CODIAD_GID=${CODIAD_GID:-2743}

if [ ! "$(id -u john)" -eq "$CODIAD_UID" ]; then
    usermod -o -u "$CODIAD_UID" john
fi
if [ ! "$(id -g john)" -eq "$CODIAD_GID" ]; then
    groupmod -o -g "$CODIAD_GID" john
fi

exec "$@"
