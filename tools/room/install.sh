#!/bin/bash
# Builds Little Guy's room on a Linux machine with systemd (blueprint sandbox-plan-v0.md). Run as root.
#   two users with no login and no sudo: littleguy-mind (him) and littleguy-guard (the other side of the mail slot),
#   sharing only the group lgmail, which is what lets mail pass through the slot
#   /opt/littleguy            his code and brain, read-only to him
#   /var/lib/littleguy/inbox  the guard writes, he reads (what his senses and his person say)
#   /var/lib/littleguy/room   a small fixed-size disk image (a diary with a set number of pages), holding
#       notebook/             his alone
#       outbox/               he drops notes, the guard reads and removes them
#   units: littleguy-guard (the guard), littleguy-mind (him; a stand-in loop until the real mind moves in),
#          littleguy-escape (the break-out test, inside exactly his walls)
# Settings (env): MEMORY (his memory cap, default 2G), CPUQUOTA (default 200% = two cores), ROOM_MB (default 64),
#   SAY_REST (seconds between spoken lines, default 7200), PERIOD (stand-in mind's seconds per thought, default 60),
#   BRAIN (a brain file to put in /opt/littleguy/brain; default: an empty placeholder)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MEMORY=${MEMORY:-2G}; CPUQUOTA=${CPUQUOTA:-200%}; ROOM_MB=${ROOM_MB:-64}
SAY_REST=${SAY_REST:-7200}; PERIOD=${PERIOD:-60}
PY=$(readlink -f /usr/bin/python3)
EXECPATHS="$PY /usr/lib -/usr/lib64 -/lib64 -/opt/littleguy/bin"

echo "== users (no login shell, no home, no sudo)"
getent group lgmail >/dev/null || groupadd --system lgmail
for u in littleguy-mind littleguy-guard; do
  id "$u" >/dev/null 2>&1 || useradd --system --user-group --no-create-home --home-dir /nonexistent \
                                     --shell /usr/sbin/nologin "$u"
  usermod -aG lgmail "$u"
done

echo "== his code and brain, read-only"
install -d -o root -g root -m 0755 /opt/littleguy
install -o root -g root -m 0644 "$HERE/guard.py" "$HERE/mind_stub.py" "$HERE/escape_test.py" /opt/littleguy/
if [ -n "${BRAIN:-}" ]; then install -o root -g root -m 0644 "$BRAIN" /opt/littleguy/brain
elif [ ! -e /opt/littleguy/brain ]; then install -o root -g root -m 0644 /dev/null /opt/littleguy/brain; fi

echo "== the mail slot and his room (a ${ROOM_MB} MB disk image)"
install -d -o root -g root -m 0755 /var/lib/littleguy /var/lib/littleguy/room
install -d -o littleguy-guard -g lgmail -m 2750 /var/lib/littleguy/inbox
IMG=/var/lib/littleguy-room.img
if [ ! -e "$IMG" ]; then
  fallocate -l "${ROOM_MB}M" "$IMG"; chmod 0600 "$IMG"; mkfs.ext4 -q -F -m 0 "$IMG"
fi
cat > /etc/systemd/system/var-lib-littleguy-room.mount <<EOF
[Unit]
Description=Little Guy's room (a fixed-size disk image)
[Mount]
What=$IMG
Where=/var/lib/littleguy/room
Type=ext4
Options=loop,nodev,nosuid,noexec
[Install]
WantedBy=local-fs.target
EOF
systemctl daemon-reload
systemctl enable --now var-lib-littleguy-room.mount
install -d -o littleguy-mind -g littleguy-mind -m 0700 /var/lib/littleguy/room/notebook
install -d -o littleguy-mind -g lgmail -m 2770 /var/lib/littleguy/room/outbox

echo "== units"
HARDEN=$(sed -e "s|@EXECPATHS@|$EXECPATHS|" -e "s|@MEMORY@|$MEMORY|" -e "s|@CPUQUOTA@|$CPUQUOTA|" \
             -e '/^#/d' -e '/^$/d' "$HERE/hardening.conf")
# inside an OrbStack machine on a Mac: hide the Mac's shared disks and its command bridge from both units
EXTRA=""
if [ -d /opt/orbstack-guest ]; then
  EXTRA=$(sed -e '/^#/d' -e '/^$/d' "$HERE/hardening-orbstack.conf")
  HARDEN="$HARDEN
$EXTRA"
  echo "== OrbStack machine: adding the Mac walls (hardening-orbstack.conf)"
fi

cat > /etc/systemd/system/littleguy-guard.service <<EOF
[Unit]
Description=Little Guy's guard (the other side of his mail slot)
RequiresMountsFor=/var/lib/littleguy/room
[Service]
ExecStart=$PY /opt/littleguy/guard.py run
Environment=SAY_REST=$SAY_REST
Restart=always
User=littleguy-guard
Group=littleguy-guard
SupplementaryGroups=lgmail
UMask=0027
StateDirectory=littleguy-guard
StateDirectoryMode=0700
RuntimeDirectory=littleguy
RuntimeDirectoryMode=0755
RuntimeDirectoryPreserve=yes
ReadWritePaths=/var/lib/littleguy/room/outbox /var/lib/littleguy/inbox
NoNewPrivileges=yes
CapabilityBoundingSet=
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
PrivateDevices=yes
PrivateNetwork=yes
IPAddressDeny=any
RestrictAddressFamilies=AF_UNIX
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectKernelLogs=yes
ProtectControlGroups=yes
ProtectClock=yes
ProtectHostname=yes
ProtectProc=invisible
ProcSubset=pid
RestrictNamespaces=yes
RestrictRealtime=yes
RestrictSUIDSGID=yes
LockPersonality=yes
MemoryDenyWriteExecute=yes
RemoveIPC=yes
SystemCallArchitectures=native
SystemCallFilter=@system-service
SystemCallFilter=~@privileged @resources
SystemCallErrorNumber=EPERM
MemoryMax=128M
TasksMax=16
$EXTRA
[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/littleguy-mind.service <<EOF
[Unit]
Description=Little Guy's mind, inside his room (stand-in loop for now)
RequiresMountsFor=/var/lib/littleguy/room
After=littleguy-guard.service
[Service]
ExecStart=$PY /opt/littleguy/mind_stub.py
Environment=PERIOD=$PERIOD
Restart=on-failure
RestartSec=30
$HARDEN
[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/littleguy-escape.service <<EOF
[Unit]
Description=Break-out test, inside exactly Little Guy's walls
RequiresMountsFor=/var/lib/littleguy/room
[Service]
Type=oneshot
# when the memory cap stops the test's memory hog, keep the rest of the test running (doesn't change his walls)
OOMPolicy=continue
ExecStart=$PY /opt/littleguy/escape_test.py /var/lib/littleguy/room/notebook/escape-results.json $(numfmt --from=iec "$MEMORY" | awk '{print int($1/1048576)}')
$HARDEN
EOF

# Inside a container (OrbStack machines are LXC), systemd adds a drop-in for EVERY service, e.g.
# /run/systemd/system/service.d/zzz-lxc-service.conf, that quietly resets the walls (ProtectSystem=no,
# PrivateNetwork=no, NoNewPrivileges=no, ReadWritePaths= ...). Drop-ins apply in file-name order, so each of his units
# gets its own that sorts after it and puts back exactly the settings it reset, copied from the unit itself.
for u in littleguy-guard littleguy-mind littleguy-escape; do
  keys=$( { cat /run/systemd/system/service.d/*.conf /etc/systemd/system/service.d/*.conf 2>/dev/null || true; } \
         | { grep -oE '^[A-Za-z]+=' || true; } | sort -u)
  rm -rf "/etc/systemd/system/$u.service.d"
  [ -z "$keys" ] && continue
  mkdir -p "/etc/systemd/system/$u.service.d"
  { echo "# puts back the walls a container drop-in reset (see install.sh)"; echo "[Service]"
    for k in $keys; do grep -E "^$k" "/etc/systemd/system/$u.service" || true; done
  } > "/etc/systemd/system/$u.service.d/zzzz-littleguy-walls.conf"
  echo "== $u: container drop-in found, walls put back ($(grep -c = "/etc/systemd/system/$u.service.d/zzzz-littleguy-walls.conf") settings)"
done

systemctl daemon-reload
systemctl enable --now littleguy-guard.service
systemctl restart littleguy-guard.service
echo "ROOM_INSTALLED (mind not started; start it with: systemctl start littleguy-mind)"
