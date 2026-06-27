#!/bin/bash

# Regenerate the autotools build system, then install git hooks.
# The .vscode / .devcontainer symlinks are set up by the plasmon-labs-devcontainer
# container repo — do not add that setup here.

libtoolize --force || glibtoolize
aclocal
autoheader --force
autoconf
automake --add-missing --copy

# Install clang-format pre-commit hook
if [ -f scripts/install_clang_format_hook.sh ]; then
    echo "Installing clang-format pre-commit hook..."
    ./scripts/install_clang_format_hook.sh
fi

# Install pre-push hook
if [ -f scripts/install_pre_push_hook.sh ]; then
    echo "Installing pre-push hook..."
    ./scripts/install_pre_push_hook.sh
fi
