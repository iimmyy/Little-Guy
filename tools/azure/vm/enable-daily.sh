#!/bin/bash
# Install a daily trigger for the existing oven build without changing VM power schedules.
# Run as root on lg-compile only when the existing build and other compute are idle.
set -euo pipefail
[[ $(id -u) == 0 ]] || { echo 'Root is required'; exit 1; }
[[ -f /opt/lg/nightly.sh && -f /etc/systemd/system/lg-nightly.service ]] || exit 1
state=$(systemctl show lg-nightly.service -p ActiveState --value)
[[ "$state" == inactive || "$state" == failed ]] || { echo "Build service is $state; refusing changes"; exit 1; }
[[ -z $(who) ]] || { echo 'Interactive sessions present; refusing changes'; exit 1; }
if ps -eo comm= | grep -Eq '^(cmake|ninja|make|cc1plus|cc1|test-backend-op|llama-server|llama-bench)$'; then
    echo 'Compute is active; refusing changes'; exit 1
fi
for p in /opt/lg/nightly-daily.sh /etc/systemd/system/lg-nightly.timer /etc/systemd/system/lg-nightly.service.d/daily.conf; do
    [[ ! -e "$p" ]] || { echo "Existing $p needs review; refusing overwrite"; exit 1; }
done
backup=/opt/lg/repair-backups/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$backup" /etc/systemd/system/lg-nightly.service.d
cp -p /etc/systemd/system/lg-nightly.service "$backup/"
cp -p /opt/lg/nightly.sh "$backup/"
systemctl is-enabled lg-nightly.service > "$backup/service-enabled.txt" || true
cat > /opt/lg/nightly-daily.sh <<'DAILY'
#!/bin/bash
set -euo pipefail
# The lock also protects direct calls to this wrapper. systemd serializes service starts.
exec 9>/opt/lg/nightly.lock
flock -n 9 || { echo 'Build already locked; skipping duplicate'; exit 0; }
day=$(date -u +%F)
marker=/opt/lg/nightly-success-$day
[[ ! -f "$marker" ]] || { echo "Build already completed for $day"; exit 0; }
[[ -z $(who) ]] || { echo 'Interactive sessions present; deferring build'; exit 75; }
if ps -eo comm= | grep -Eq '^(cmake|ninja|make|cc1plus|cc1|test-backend-op|llama-server|llama-bench)$'; then
    echo 'Other compute is active; deferring build'; exit 75
fi
/opt/lg/nightly.sh
# The original runner can exit zero after a failed stage; require its actual final result.
last_result=$(grep 'nightly done:' "/opt/lg/nightly-$day.log" | tail -1)
[[ "$last_result" == *'nightly done: ok' ]] || { echo 'Build did not finish successfully'; exit 1; }
printf '%s\n' "$(date -u +%FT%TZ)" > "$marker"
echo "Daily build success recorded for $day"
DAILY
chmod 755 /opt/lg/nightly-daily.sh
cat > /etc/systemd/system/lg-nightly.service.d/daily.conf <<'UNIT'
[Service]
ExecStart=
ExecStart=/opt/lg/nightly-daily.sh
Nice=10
UNIT
cat > /etc/systemd/system/lg-nightly.timer <<'TIMER'
[Unit]
Description=Daily Little Guy oven build while the VM remains running

[Timer]
OnCalendar=*-*-* 00:35:00 UTC
Persistent=true
Unit=lg-nightly.service

[Install]
WantedBy=timers.target
TIMER
bash -n /opt/lg/nightly-daily.sh
systemd-analyze verify /etc/systemd/system/lg-nightly.timer /etc/systemd/system/lg-nightly.service
systemctl daemon-reload
# This activates only the new build timer; no VM wake/shutdown configuration is touched.
systemctl enable --now lg-nightly.timer
systemctl start --no-block lg-nightly.service
printf 'BACKUP=%s\n' "$backup"
systemctl show lg-nightly.service -p ActiveState -p SubState -p ExecStart
systemctl list-timers lg-nightly.timer --all --no-pager
