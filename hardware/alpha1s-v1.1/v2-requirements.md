# V2 requirements

**Superseded on 2026-10-21:** the user selected the [newer servo-adapter plan](../little-guy-v2/README.md). The decisions below document the earlier Alpha-derived approach. Jetson-hosted AI and a user-selected speaker carry forward; an Alpha mainboard copy is no longer required for the selected V2.

User decisions recorded 20 October 2026. These describe the planned V2, not verified features of the original board.

- The board connects to a Jetson Orin Nano Super. The AI model runs on the Jetson and controls the board.
- V2's purpose is to improve the recovered Alpha board specifically for use with the Jetson. Prioritize the host interface, predictable motion execution, feedback and power integration. Original-board reconstruction provides the reference; V2 does not need to retain every original feature or component. Specific removals or replacements remain design decisions, not already authorized selections.
- The user will supply their own speaker. Its model, impedance, power rating, dimensions and passive/active type are not yet specified. Matching the original speaker is not a V2 requirement.
- The existing objective remains reconstruction of the original Alpha 1S V1.1 PCB and circuit as the basis for V2.

## Proposed division of responsibilities

The Jetson runs AI inference and application logic. A host-side controller translates requested actions into explicit, bounded motion commands. The board's microcontroller handles servo-bus communication, command timing, available feedback and local fault handling. These are proposed responsibilities; firmware and donor interfaces have not been verified.

Use one bidirectional wired command/status link. USB serial is the first option to evaluate; UART is another option after voltage levels and the exact carrier are verified. No connector, pin assignment, baud rate or protocol is selected yet. The donor's USB connector does not by itself establish a usable control interface.

V2 firmware should validate commands and joint limits, detect expired commands or a lost host, and implement a mechanically appropriate fault response. Whether that response holds position, stops a trajectory or disables outputs must be decided from the robot's mechanics; immediate torque removal is not assumed safe. Distinguish commanded targets from measured servo feedback.

Jetson power must be designed for its actual carrier and operating load. Existing Alpha power circuitry has not been shown to supply it. Keep power integration unresolved until the donor rails, battery and Jetson configuration are known.

Speaker playback may use the board's audio path or a separate Jetson-connected audio device. Select that path after the user's speaker is specified; no amplifier or speaker connector change is made yet.

## Remaining decisions

Exact Jetson carrier/dev-kit configuration; wired link and firmware strategy; available servo feedback; power topology; speaker specification and audio path. These do not resolve the missing donor connections, component values or physical land dimensions.

NVIDIA reference: [Orin Nano developer-kit hardware layout](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/hardware_layout.html). Carrier capabilities do not establish compatibility with the donor board.
