# Audit of supplied Alpha 1S research report

The supplied report is preserved at `reference/submitted-research/deep-research-report.md` as unverified secondary material. It contains no source URLs, numbered references, actual donor netlist, or PCB fabrication data. Its generic engineering assumptions do not establish an identical Alpha1S V1.1 board.

| Report claim | Current evidence and disposition |
| --- | --- |
| Approximately100x80mm board | Superseded by user caliper readings:72.08mm maximum width,68.45mm length,56.15mm neck width,1.64mm thickness. See geometry.json. |
| Four layers and35um copper | Unestablished. Visible surfaces and board thickness do not determine layer count, dielectric stack or copper weight. Do not import as donor stackup. |
| STM32F103RDT6 | Supported by direct donor reading and manufacturer documentation. U8 has a64-pin10x10mm LQFP package. Its reference footprint pitch is0.50mm, contradicting the report manufacturing section0.65mm. See chip-readings and mcu-pcb-placement records. |
| Nordic/TI generic Bluetooth or Wi-Fi module | Donor U10 is labelled BM77SPP03MC2. Do not substitute unidentified radios or infer Wi-Fi. |
| AMS1117/MP1410 regulator examples | Donor U13/U14 identified as SGM2019-3.3; U3 BQ24133 charger; U1 remains unidentified. These examples are not donor BOM entries or verified replacements. |
| Unspecified speaker amplifier | Donor U12 is PAM8003; U4 is MS6324G DAC. Speaker impedance and exact donor audio wiring still need evidence. |
| Battery2pins | Donor J2 has four visible contacts. Their assignments are unresolved. Do not collapse the connector to a generic two-pin symbol. |
| Servo+5V and connector pinouts | Unverified for this donor. Public servo research does not establish mainboard contacts, rail voltage, or MCU endpoints. |
| External EEPROM, IMU,32.768kHz crystal | Not established by this report. No corresponding parts added. |
| Firmware can be backed up through factory USB bootloader | Incorrect for STM32F103RD factory ROM. High-density STM32F103 uses USART1. A USB update protocol could instead be provided by application/vendor firmware; donor behavior is unverified. |
| SWD necessarily permits firmware dump | Conditional on access and protection. Read protection blocks normal debug flash reads. Removing read protection causes main flash mass erase. Donor protection status is unknown. |
| Generic supplier numbers/prices and substitutions | Not validated as orderable donor BOM. No purchase or substitution authorized by this report. |
| Timeline marks identification complete | Does not reflect project state. U1, many passives, most connections and geometry remain unresolved. |

ST primary sources checked2026-10-20:

- [AN2586 Rev8, section4.3](https://www.st.com/resource/en/application_note/cd00164185-getting-started-with-stm32f10xxx-hardware-development-stmicroelectronics.pdf): high-density STM32F10xxx factory bootloader uses USART1; USB DFU support described for the different connectivity-line devices.
- [PM0075 Rev2, sections2.3.5 and2.4.1](https://www.st.com/resource/en/programming_manual/pm0075-stm32f10xxx-flash-memory-microcontrollers-stmicroelectronics.pdf): read protection and mass erase on transition to unprotected state.

These documents describe MCU behavior, not the donor board debug-pad wiring or protection configuration. No hardware/debug commands, option-byte changes, erase, or flash operations were performed.

The report contributes a general workflow checklist but no new verified mainboard connections. Its instruction to insert typical circuits would change a reconstruction into a reference design. Existing evidence-led schematic and PCB remain authoritative; objective remains identical original-board reconstruction before a separate V2.
