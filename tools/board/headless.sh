#!/bin/bash
# Headless board: switch off what a screenless, speakerless, wired box never uses, so nothing wakes the cores
# or holds memory behind the brain's back. Every change is listed here and undone by `undo`.
#   ssh <board> 'sudo bash -s apply'  < tools/board/headless.sh
#   ssh <board> 'sudo bash -s undo'   < tools/board/headless.sh
#   ssh <board> 'bash -s status'      < tools/board/headless.sh
# Takes full effect after a reboot (the GPU driver option is read when the driver loads).

# system services: desktop helpers, Bluetooth, disk automounting, battery, firmware updater, a domain controller,
# the login screen on the (absent) monitor, and Ubuntu Pro. Masked so nothing can start them on demand either.
SERVICES="bluetooth.service fwupd.service udisks2.service upower.service rtkit-daemon.service samba-ad-dc.service
          ubuntu-advantage.service ua-reboot-cmds.service getty@tty1.service"
# the ones that were enabled at boot before (restored by undo)
WERE_ENABLED="bluetooth.service udisks2.service upower.service samba-ad-dc.service ubuntu-advantage.service
              ua-reboot-cmds.service getty@tty1.service"
# timers that wake the board for updates, news and manual-page indexing; the system updater could even reboot it
# or swap the kernel under us. Updates now happen by hand.
TIMERS="fwupd-refresh.timer motd-news.timer man-db.timer systemd-sysupdate.timer systemd-sysupdate-reboot.timer
        apt-daily.timer apt-daily-upgrade.timer"
# per-user desktop services: the sound server, Bluetooth file transfer, the password keyring and the file indexer
USER_UNITS="pipewire.service pipewire.socket pipewire-pulse.service pipewire-pulse.socket wireplumber.service
            filter-chain.service obex.service gnome-keyring-daemon.service gnome-keyring-daemon.socket
            gcr-ssh-agent.service gcr-ssh-agent.socket tracker-miner-fs-3.service session-migration.service"
# the GPU gets its own device, so the display side can stay off without taking the GPU with it
MODPROBE=/etc/modprobe.d/littleguy-headless.conf

case "$1" in
apply)
    systemctl disable --now $TIMERS 2>/dev/null
    systemctl mask --now $SERVICES 2>/dev/null
    systemctl --global mask $USER_UNITS 2>/dev/null
    for u in $(loginctl list-users --no-legend | awk '{print $2}'); do
        systemctl --user -M "$u@" stop $USER_UNITS 2>/dev/null
    done
    nmcli radio wifi off   # wired only; the radio was scanning with nothing to join
    echo "options msm separate_gpu_kms=1" > $MODPROBE
    echo "applied; reboot for the GPU split" ;;
undo)
    rm -f $MODPROBE
    nmcli radio wifi on
    systemctl --global unmask $USER_UNITS 2>/dev/null
    systemctl unmask $SERVICES 2>/dev/null
    systemctl enable $WERE_ENABLED 2>/dev/null
    systemctl enable --now $TIMERS 2>/dev/null
    echo "undone; reboot to bring everything back" ;;
status)
    for u in $SERVICES $TIMERS; do printf '%-34s %s\n' "$u" "$(systemctl is-enabled $u 2>/dev/null)"; done
    for u in $USER_UNITS; do printf '%-34s %s (user)\n' "$u" "$(systemctl --global is-enabled $u 2>/dev/null)"; done
    echo "wifi radio: $(nmcli radio wifi)"
    echo "gpu option: $(cat $MODPROBE 2>/dev/null || echo none)" ;;
*)  echo "usage: headless.sh apply|undo|status"; exit 1 ;;
esac
