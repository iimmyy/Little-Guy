# Headless Android runtime (WIP)

I want Little Guy's runtime to exist independently of a screen. Android calls the screen an `Activity`; it can be destroyed whenever I leave the app, rotate the device, or remove its task. That makes it the wrong owner for the robot's brain and hardware.

I moved runtime ownership into `LittleGuyService`, a started and bound foreground service:

- `LittleGuyService` owns Gemma, conversation memory, `RobotCoordinator`, the safety gate, and the hardware backend.
- `MainActivity` starts and binds to the service, sends messages through its local binder, and renders service snapshots.
- Closing the dashboard removes its observer and binding but does not stop the service.
- The service keeps only the latest 250 transcript lines so a returning dashboard can recover useful state without unbounded memory growth.
- Android 15 requires a declared foreground-service purpose. The current build uses `specialUse` for the continuous on-device companion-robot runtime. I will add narrower hardware service types when the corresponding hardware actually exists.

## What I verified

I built the app with Homebrew OpenJDK 17, ran the unit tests, and installed the debug APK on the Android 15 ARM64 emulator.

The first launch exposed a real Android 15 failure: the service crashed because it had no declared foreground-service type. I added the `specialUse` type, its permission, and its required subtype description, then repeated the test.

After the correction:

- Android reported `LittleGuyService` as a foreground service.
- Pressing Home unbound the dashboard while the process kept the same PID.
- Swiping the dashboard task out of Recents removed the activity task while the service kept the same PID.
- Reopening the dashboard was a warm launch and reconnected to the same service process.
- The service found the existing private Gemma model without initializing a second engine.
- Android lint completed with no errors. Two pre-existing packaging warnings remain for backup rules and the temporary launcher icon.

## Physical Q6A bring-up

I installed the app on the Q6A and added a `BOOT_COMPLETED` receiver for the current lightweight `specialUse` service. After a real reboot, the Mac reconnected over persistent TCP ADB and Android reported `LittleGuyService` running in the foreground with `BOOT_COMPLETED` as its start reason. The dashboard was not opened. I subsequently copied the model and verified two responses in the running service. This proves boot startup and initial inference, not recovery after Android kills the process.

The board uses non-root ADB on LAN port 5555 because paired Wireless debugging did not stay enabled across reboot on this image. That port is reachable by other devices on the LAN; it is a bring-up convenience, not a finished security configuration. The Mac connection helper caches the current address only in Git-ignored `.run/q6a-host`.

This boot receiver is not permission to add always-on camera or microphone capture. Android 15 restricts those foreground-service types at boot. Before the robot gains those sensors, I still need dedicated-device provisioning, explicit sensor lifecycle, screen-off and crash-restart tests, and a recovery path. The factory Qualcomm test apps will stay installed until model, USB, camera, audio, and thermal baselines pass; any later debloat must be reversible.

## Local build command

Android Studio currently bundles JDK 25, which this Gradle setup does not accept. Homebrew OpenJDK 17 is already installed, so I build with:

```sh
JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
  ./gradlew testDebugUnitTest assembleDebug lintDebug
```
