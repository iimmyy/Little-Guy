# First bench test: temperature permission and motor-arm latch

2026-10-25. **Prepared procedure; not performed.** Owner has no bench equipment available at present.
No equipment purchase is requested yet. Continue design/package work; finalize a small setup list when this test
is ready to schedule. This checks two low-voltage circuits from the power schematic.
It does not qualify the charger, battery pack, motor supply, switches or a robot stop strategy.

Use only a regulated **3.3 V** supply with a **30 mA current limit** for this test. Leave the battery, USB-C charger,
wireless receiver, Jetson and every servo disconnected. No high-current switching circuit is part of this setup.
The first practical step is confirming the available supply, meter and IC adapters before buying test parts.

## Parts from the circuit

| Circuit | Parts to fit for this test |
|---|---|
| Three sensor windows | U1201/U1202 TLV1704PWR on TSSOP-14-to-breadboard adapters; R1201 19.1k 0.1%, R1202 10k 0.1%, R1203 10k 0.1%, R1204 26.1k 0.1%, R1205 10k 1%; three R1206-R1208 10k 0.1%; C1201-C1205 100n |
| Charge permission | U303 SN74LVC1G00DBVR on a SOT-23-5 adapter; C312 100n; R307 100k from PACK_TEMP_OK to ground. Use a 100k pull-up from CE_N to the bench 3.3 V for this isolated test. CHG_ALLOW must have a 100k pull-down and a jumper to 3.3 V. |
| Motor-arm latch | U1001 SN74LVC74APWR on a TSSOP-14 adapter, U1003 TLV803EA30DBZR on a SOT-23 adapter, R1001 3.3k, R1002/R1003 100k, C1001/C1003 100n; two jumpers or switches |
| Sensor substitutes | Three 10k resistors for room temperature, one 27.4k resistor for a cold substitute, one 4.12k for hot, plus open/short jumpers. These test circuit states; they are not a thermistor calibration. |

The SMT values in the production schematic can be represented by equivalent through-hole resistors for this bench
test. Use the actual listed ICs. Adapter pin numbering must be checked against the IC manufacturer's top-view pinout
before power is applied; do not assume every breakout labels pads in package order. No bench-adapter order is placed.

## Temperature-window wiring

Follow power sheet `12-pack-temperature` and `03-housekeeping`. Connect AON_3V3 to the bench 3.3 V and GND to
the bench return. All three NTC inputs have a 10k pull-up to 3.3 V; substitute resistors go from each NTC input to GND.

TLV1704 pin map: supply pin 3 to 3.3 V, pin 12 to GND. First IC inputs 4/5, 6/7, 8/9, 10/11 are the four comparators;
outputs 2/1/14/13 respectively. Connect these as shown in the schematic. On the second IC, use its first two
comparators for segment 3; tie unused negative inputs 8/10 to 3.3 V, positive inputs 9/11 to GND and leave outputs
14/13 unconnected. Six used open-collector outputs share PACK_TEMP_OK; do not connect this node to a push-pull output.

For the NAND gate: pin 5=3.3 V, pin 3=GND, pin 1=CHG_ALLOW, pin 2=PACK_TEMP_OK, pin 4=CE_N.
The real board uses a separate charger-REGN pull-up; **do not connect REGN or a charger for this test**.

With all three sensor substitutes at 10k, measure:

| Point | Nominal reading / criterion |
|---|---|
| NTC_SEG1/2/3 | About 1.65 V each |
| TEMP_HOT_REF | About 1.134 V |
| TEMP_COLD_REF | About 2.386 V |
| PACK_TEMP_OK | About 3.00 V with the 10k pull-up and 100k pull-down |
| CE_N with CHG_ALLOW low | Logic high; charging inhibited |
| CE_N with CHG_ALLOW high and all sensors in range | Logic low; permission granted |

Then replace **one sensor at a time**, leaving the other two at 10k. Check every row for all three sensor inputs.

| Substitute on the selected input | Expected PACK_TEMP_OK | Expected CE_N with CHG_ALLOW high |
|---|---|---|
| Open circuit | Low | High |
| Short to GND | Low | High |
| 27.4k, cold substitute | Low | High |
| 4.12k, hot substitute | Low | High |
| Return to 10k | High | Low |

Also disconnect the CHG_ALLOW high jumper: its pull-down must return CE_N high. Record measured voltages and
supply current. If a state is wrong, leave charging disabled and repair the circuit before proceeding. A multimeter
checks static states; an oscilloscope is needed later to check chatter, startup and short faults. Actual thermistor
tolerance, thermal attachment and temperature trip points remain unvalidated after this resistor-substitution test.

## Motor latch wiring and test

On SN74LVC74A: pin 14=3.3 V, 7=GND, 2=3.3 V, 4=3.3 V; pin 1=MOTOR_CLEAR_N, 3=MOTOR_ARM, 5=MOTOR_ARMED.
Leave pin 6 unconnected. Hold the unused half reset: pins 11/12/13=GND, pin 10=3.3 V, pins 8/9 unconnected.
Add a 100n bypass at the supply pins. MOTOR_ARM has its 100k pull-down; a momentary jumper to 3.3 V supplies an arm edge.

Connect 3.3 V through R1001=3.3k to CUT_LOOP_FEED. A normally closed switch/jumper connects CUT_LOOP_FEED to
MOTOR_CLEAR_N. R1002=100k connects MOTOR_CLEAR_N to GND. Supervisor U1003 pin 3=3.3 V, pin 1=GND,
pin 2=MOTOR_CLEAR_N. Its 3.0 V threshold and 200 ms startup delay must be verified against the actual ordered suffix.

1. Power on with the cut loop closed and ARM low. MOTOR_ARMED must remain low after startup.
2. Apply one ARM rising edge. MOTOR_ARMED should go high.
3. Open the cut loop. MOTOR_ARMED must go low.
4. Reclose the cut loop without a new ARM edge. MOTOR_ARMED must stay low.
5. Pulse ARM again. It may go high now.
6. Repeat after power removal/reapplication. It must not automatically arm.
7. Later, add the real spinal watchdog to the same clear node and verify a stalled heartbeat clears this latch.

Record supply, CLEAR_N, ARM and Q voltages plus any unexpected state. This test deliberately measures the latch
output; it does not energize motors. Brownout thresholds/timing need a controlled supply sweep and scope before release.

## Results to collect

Equipment and supply setting; actual IC suffixes and adapter pin check; the table readings for all three inputs;
all seven latch steps; any unstable transitions. Results are currently **not collected**.

Passing this first test is one gate. Power-stage transient/thermal tests, returned-energy handling, source handover,
BMS protection and charge-profile tests are separate work and are still required before board layout release.
