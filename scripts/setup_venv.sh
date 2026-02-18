#!/bin/bash
#
# setup_venv.sh — Create a shared Python virtual environment for ZoneMinder
#
# Usage:
#   sudo ./scripts/setup_venv.sh                          # base only (ZM API client)
#   sudo ./scripts/setup_venv.sh --extras ml              # + ML detection
#   sudo ./scripts/setup_venv.sh --extras ml,serve        # + ML + remote server
#   sudo ./scripts/setup_venv.sh --extras ml,serve,train  # everything
#   sudo ZM_VENV=/custom/path ./scripts/setup_venv.sh
#
# NOTE: The base install (no --extras) only includes the ZM API client.
#       For ML detection (Detector), add the 'ml' extra.
#       The 'serve' and 'train' extras auto-include ML dependencies.
#
# This script:
#   1. Installs the python3-venv OS package if missing
#   2. Creates a venv at ZM_VENV (default: /opt/zoneminder/venv)
#   3. Installs pyzm into it
#
# The same venv is shared with zmeventnotification's install.sh so both
# projects use a single, isolated Python environment.

set -euo pipefail

ZM_VENV="${ZM_VENV:-/opt/zoneminder/venv}"
ZM_VENV_OWNER="${ZM_VENV_OWNER:-www-data}"
EXTRAS=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv-path)  ZM_VENV="$2";       shift 2 ;;
        --owner)      ZM_VENV_OWNER="$2"; shift 2 ;;
        --extras)     EXTRAS="$2";         shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
print_info()    { echo -e "\033[1;36m[INFO]\033[0m $1"; }
print_success() { echo -e "\033[1;32m[OK]\033[0m $1"; }
print_error()   { echo -e "\033[1;31m[ERROR]\033[0m $1"; }

detect_os() {
    if [[ "$(uname -s)" == "Darwin" ]]; then
        echo "macos"
    elif command -v apt-get &>/dev/null; then
        echo "debian"
    elif command -v dnf &>/dev/null; then
        echo "fedora"
    elif command -v yum &>/dev/null; then
        echo "centos"
    else
        echo "unknown"
    fi
}

# ---------------------------------------------------------------------------
# Ensure python3 -m venv works
# ---------------------------------------------------------------------------
ensure_venv_package() {
    if python3 -m venv --help &>/dev/null; then
        return 0
    fi

    local os
    os=$(detect_os)
    print_info "python3-venv not available — installing for ${os}..."

    case "$os" in
        debian)
            apt-get update -qq && apt-get install -y -qq python3-venv
            ;;
        fedora)
            dnf install -y python3-libs  # venv is bundled in python3-libs on Fedora
            ;;
        centos)
            yum install -y python3-libs
            ;;
        macos)
            # venv is always available with python.org or Homebrew Python
            print_error "python3 -m venv failed on macOS. Ensure you have a full Python install (brew install python3)."
            exit 1
            ;;
        *)
            print_error "Cannot auto-install python3-venv. Install it manually, then re-run."
            exit 1
            ;;
    esac

    if ! python3 -m venv --help &>/dev/null; then
        print_error "python3-venv still not working after install attempt."
        exit 1
    fi
    print_success "python3-venv installed"
}

# ---------------------------------------------------------------------------
# Create or reuse the venv
# ---------------------------------------------------------------------------
create_venv() {
    if [[ -d "${ZM_VENV}" && -x "${ZM_VENV}/bin/python" ]]; then
        print_info "Venv already exists at ${ZM_VENV} — reusing"
        return 0
    fi

    print_info "Creating venv at ${ZM_VENV} ..."
    mkdir -p "$(dirname "${ZM_VENV}")"
    python3 -m venv --system-site-packages "${ZM_VENV}"
    print_success "Venv created (Python: $(${ZM_VENV}/bin/python --version))"
}

# ---------------------------------------------------------------------------
# Shim opencv-python if cv2 is already available (source/system build)
# ---------------------------------------------------------------------------
shim_opencv() {
    local venv_python="${ZM_VENV}/bin/python"

    # Can the venv Python already import cv2?
    local cv2_version
    cv2_version=$("${venv_python}" -c "import cv2; print(cv2.__version__)" 2>/dev/null) || return 0

    # Is there already a pip-registered opencv-python? If so, no shim needed.
    if "${venv_python}" -m pip show opencv-python &>/dev/null; then
        return 0
    fi

    local site_packages
    site_packages=$("${venv_python}" -c "import sysconfig; print(sysconfig.get_path('purelib'))")

    local dist_dir="${site_packages}/opencv_python-${cv2_version}.dist-info"
    mkdir -p "${dist_dir}"

    cat > "${dist_dir}/METADATA" <<EOF
Metadata-Version: 2.1
Name: opencv-python
Version: ${cv2_version}
Summary: Shim — real cv2 is provided by a source/system build
EOF

    echo > "${dist_dir}/RECORD"
    echo "opencv-python" > "${dist_dir}/top_level.txt"
    echo "Wheel-Version: 1.0" > "${dist_dir}/WHEEL"

    print_success "opencv-python shim created (cv2 ${cv2_version} from source/system)"
}

# ---------------------------------------------------------------------------
# Install pyzm
# ---------------------------------------------------------------------------
install_pyzm() {
    local pip="${ZM_VENV}/bin/pip"
    local pkg="."

    if [[ -n "${EXTRAS}" ]]; then
        pkg=".[${EXTRAS}]"
    fi

    # If we're running from the pyzm repo root, install in editable/local mode
    # Otherwise install from PyPI
    if [[ -f "setup.py" || -f "pyproject.toml" ]]; then
        print_info "Installing pyzm from local source (${pkg}) ..."
        "${pip}" install --upgrade pip setuptools wheel -q
        "${pip}" install "${pkg}"
    else
        print_info "Installing pyzm from PyPI (${pkg}) ..."
        "${pip}" install --upgrade pip setuptools wheel -q
        if [[ -n "${EXTRAS}" ]]; then
            "${pip}" install "pyzm[${EXTRAS}]"
        else
            "${pip}" install pyzm
        fi
    fi
    print_success "pyzm installed into ${ZM_VENV}"
}

# ---------------------------------------------------------------------------
# Set ownership (Linux only — macOS dev boxes typically don't need this)
# ---------------------------------------------------------------------------
set_ownership() {
    if [[ "$(uname -s)" == "Darwin" ]]; then
        return 0
    fi

    if id "${ZM_VENV_OWNER}" &>/dev/null; then
        print_info "Setting ownership to ${ZM_VENV_OWNER} ..."
        chown -R "${ZM_VENV_OWNER}:" "${ZM_VENV}"
        print_success "Ownership set"
    else
        print_info "User '${ZM_VENV_OWNER}' not found — skipping chown (set ZM_VENV_OWNER if needed)"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print_info "ZM venv path: ${ZM_VENV}"
print_info "Python: $(python3 --version)"
ensure_venv_package
create_venv
shim_opencv
install_pyzm
set_ownership

echo
print_success "Done. Use this Python for ZoneMinder scripts:"
echo "  ${ZM_VENV}/bin/python"
echo
echo "To activate the venv in your shell:"
echo "  source ${ZM_VENV}/bin/activate"
