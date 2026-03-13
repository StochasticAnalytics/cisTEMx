#!/bin/bash

# Regenerate the autotools build system, then install git hooks.
#
# .vscode is a symlink into plasmon-labs-devcontainer/.vscode_shared and
# holds tasks.json (the compile-code engine's source of build-dir names and
# configure flags). The container build wires the symlink via that repo's
# setup_links.sh, but a worktree checkout or an interrupted container setup
# can leave it missing — re-establish it here so a downstream compile does
# not fail on missing tasks.json.
if [ ! -e .vscode ] && [ -x plasmon-labs-devcontainer/setup_links.sh ]; then
    echo "Re-establishing .vscode symlink via plasmon-labs-devcontainer/setup_links.sh..."
    ./plasmon-labs-devcontainer/setup_links.sh
fi

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
