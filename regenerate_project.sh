#!/bin/bash

libtoolize --force || glibtoolize
aclocal
autoheader --force
autoconf
automake --add-missing --copy

# Install clang-format-14 pre-commit hook
if [ -f scripts/install_clang_format_hook.sh ]; then
    echo "Installing clang-format-14 pre-commit hook..."
    ./scripts/install_clang_format_hook.sh
fi

# Install pre-push hook
if [ -f scripts/install_pre_push_hook.sh ]; then
    echo "Installing pre-push hook..."
    ./scripts/install_pre_push_hook.sh
fi


