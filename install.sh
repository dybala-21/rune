#!/bin/sh
# RUNE installer — https://github.com/dybala-21/rune
# codename: dybala
#
# Usage:
#   curl -LsSf https://raw.githubusercontent.com/dybala-21/rune/main/install.sh | sh
#   wget -qO- https://raw.githubusercontent.com/dybala-21/rune/main/install.sh | sh
#
# Commands (pass as first argument):
#   (none)      Install RUNE (default)
#   update      Update RUNE to the latest version
#   uninstall   Remove RUNE completely
#
# Examples:
#   curl -LsSf .../install.sh | sh                  # install
#   curl -LsSf .../install.sh | sh -s -- update     # update
#   curl -LsSf .../install.sh | sh -s -- uninstall  # uninstall
#   rune self update                                 # update (after install)
#   rune self uninstall                              # uninstall (after install)
#
# Environment variables:
#   RUNE_VERSION=0.1.0     — install a specific version (default: latest)
#   RUNE_EXTRAS=full       — extras to install (default: full)
#                            Options: full, vector, browser, embedding
#                            Use "none" for minimal core-only install
#
# License: MIT

set -eu

# --- Configuration -----------------------------------------------------------

RUNE_REPO="https://github.com/dybala-21/rune.git"
RUNE_PACKAGE="rune-ai"
RUNE_PYTHON="python3.13"
RUNE_EXTRAS="${RUNE_EXTRAS:-full}"
UV_INSTALL_URL="https://astral.sh/uv/install.sh"

# --- Helpers -----------------------------------------------------------------

info() {
    printf '\033[1;34m==>\033[0m \033[1m%s\033[0m\n' "$1"
}

success() {
    printf '\033[1;32m==>\033[0m \033[1m%s\033[0m\n' "$1"
}

warn() {
    printf '\033[1;33mwarning:\033[0m %s\n' "$1" >&2
}

die() {
    printf '\033[1;31merror:\033[0m %s\n' "$1" >&2
    exit 1
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

ensure_path() {
    case ":${PATH}:" in
        *:"${HOME}/.local/bin":*) ;;
        *) export PATH="${HOME}/.local/bin:${PATH}" ;;
    esac
}

# --- Commands ----------------------------------------------------------------

do_install() {
    # 1. Check for a downloader
    if command_exists curl; then
        DOWNLOADER="curl"
    elif command_exists wget; then
        DOWNLOADER="wget"
    else
        die "curl or wget is required but neither was found"
    fi

    info "Installing RUNE..."

    # 2. Install uv if not present
    if command_exists uv; then
        info "uv found: $(uv --version)"
    else
        info "Installing uv..."
        if [ "$DOWNLOADER" = "curl" ]; then
            curl -LsSf "$UV_INSTALL_URL" | sh
        else
            wget -qO- "$UV_INSTALL_URL" | sh
        fi

        UV_BIN="${HOME}/.local/bin"
        if [ -f "${UV_BIN}/env" ]; then
            # shellcheck disable=SC1091
            . "${UV_BIN}/env"
        elif [ -d "${UV_BIN}" ]; then
            export PATH="${UV_BIN}:${PATH}"
        fi

        command_exists uv || die "uv installation failed. Restart your shell and re-run."
        success "uv installed"
    fi

    # 3. Build install spec (from Git, not PyPI)
    GIT_SPEC="rune-ai @ git+${RUNE_REPO}"

    if [ "$RUNE_EXTRAS" = "none" ]; then
        INSTALL_SPEC="$GIT_SPEC"
        info "Installing RUNE from GitHub (core only)..."
    else
        INSTALL_SPEC="${RUNE_PACKAGE}[${RUNE_EXTRAS}] @ git+${RUNE_REPO}"
        info "Installing RUNE from GitHub [${RUNE_EXTRAS}]..."
    fi

    # 4. Install from Git
    uv tool install --force --python "$RUNE_PYTHON" "$INSTALL_SPEC"

    # 5. Playwright browser binaries (if applicable)
    case "$RUNE_EXTRAS" in
        *browser*|*full*)
            info "Installing Chromium for browser automation..."
            uv tool run --from "$INSTALL_SPEC" playwright install chromium 2>/dev/null || \
                warn "Playwright browser install failed — run 'playwright install chromium' manually"
            ;;
    esac

    # 6. Browser extension (extract to ~/.rune/extension/)
    ensure_path
    case "$RUNE_EXTRAS" in
        *browser*|*full*)
            info "Setting up browser extension..."
            rune browser setup 2>/dev/null || \
                warn "Extension setup skipped — run 'rune browser setup' manually after install"
            ;;
    esac

    # 7. Verify
    ensure_path
    if command_exists rune; then
        echo ""
        success "RUNE installed successfully!"
        echo ""
        echo "  Version:  $(rune --version 2>/dev/null || echo 'unknown')"
        echo "  Location: $(command -v rune)"
        echo ""
        echo "  Get started:"
        echo "    rune env set OPENAI_API_KEY sk-..."
        echo "    rune"
        echo ""
        echo "  Later:"
        echo "    rune self update      # update to latest"
        echo "    rune self uninstall   # remove completely"
        echo ""
    else
        warn "Installation completed but 'rune' not found in PATH."
        echo ""
        echo '  export PATH="$HOME/.local/bin:$PATH"'
        echo "  source ~/.bashrc  # or ~/.zshrc"
    fi
}

do_update() {
    command_exists uv || die "uv not found. Run the install script first."

    ensure_path

    OLD_VERSION="unknown"
    if command_exists rune; then
        OLD_VERSION="$(rune --version 2>/dev/null || echo 'unknown')"
    fi

    info "Updating RUNE from GitHub..."
    uv tool install --force --python "$RUNE_PYTHON" "${RUNE_PACKAGE} @ git+${RUNE_REPO}"

    NEW_VERSION="unknown"
    if command_exists rune; then
        NEW_VERSION="$(rune --version 2>/dev/null || echo 'unknown')"
    fi

    # Update browser extension if present
    rune browser setup 2>/dev/null || true

    if [ "$OLD_VERSION" = "$NEW_VERSION" ]; then
        success "Already up to date: ${NEW_VERSION}"
    else
        success "Updated: ${OLD_VERSION} → ${NEW_VERSION}"
    fi
}

do_uninstall() {
    command_exists uv || die "uv not found. Nothing to uninstall."

    info "Uninstalling RUNE..."
    uv tool uninstall "$RUNE_PACKAGE" 2>/dev/null || true

    ensure_path
    if command_exists rune; then
        warn "'rune' command still found at $(command -v rune) — may be from another install method"
    else
        success "RUNE uninstalled"
        echo ""
        echo "  Your data in ~/.rune/ was NOT removed."
        echo "  To delete it: rm -rf ~/.rune"
    fi
}

# --- Main (wrapped in function to prevent partial-download execution) --------

main() {
    COMMAND="${1:-install}"

    case "$COMMAND" in
        install)    do_install ;;
        update)     do_update ;;
        uninstall)  do_uninstall ;;
        -h|--help|help)
            echo "Usage: install.sh [install|update|uninstall]"
            echo ""
            echo "Commands:"
            echo "  install     Install RUNE (default)"
            echo "  update      Update to latest version"
            echo "  uninstall   Remove RUNE"
            echo ""
            echo "Environment variables:"
            echo "  RUNE_VERSION=0.1.0   Install specific version"
            echo "  RUNE_EXTRAS=full     Extras: full, vector, browser, embedding, none"
            ;;
        *)
            die "Unknown command: $COMMAND (use install, update, or uninstall)"
            ;;
    esac
}

main "$@"
