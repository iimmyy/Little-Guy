#!/bin/bash
# One-time setup on the compile VM (sent by stage2.sh through az vm run-command): build tools, the Azure CLI, and the
# boot service that runs the nightly check-up. Arguments: the storage account name, then nightly.sh as base64.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q build-essential cmake ninja-build git gcc-aarch64-linux-gnu g++-aarch64-linux-gnu curl ca-certificates
command -v az >/dev/null || curl -sL https://aka.ms/InstallAzureCLIDeb | bash
mkdir -p /opt/lg
echo "$1" > /opt/lg/storage-account
echo "$2" | base64 -d > /opt/lg/nightly.sh
chmod 755 /opt/lg/nightly.sh
cat > /etc/systemd/system/lg-nightly.service <<'EOF'
[Unit]
Description=Little Guy nightly oven check-up
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
ExecStart=/opt/lg/nightly.sh
TimeoutStartSec=6h

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable lg-nightly.service
echo "vm ready"
