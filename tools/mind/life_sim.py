#!/usr/bin/env python3
"""A few simulated days of his life: feelings from lived time (blueprint 3.2 pacemaker, 3.3 drives, 3.4 core affect,
3.6 notebook). Immy: all emotions, not one; he is never told how he feels.

Every tick of the clock:
  senses  -> FACTS only (the time, how long since his person left, who is nearby, light, sound, battery, anything
             new on the desk); never a feeling
  drives  -> build and drain with time and events: social (time without his person), boredom (time since anything
             new), tiredness (low battery, time awake)
  feeling -> the two core dials (valence, arousal) as a state with momentum (Immy: feelings can spiral, and they
             fade, and sleep resets them): each tick = what is left of the last one after fading (half-life per
             dial) + what the drives and events pour in (AFFECT, the design decision) + what his own last thought
             pours in (rumination: how pleasant / keyed up his note reads inside him, times GAIN). Charging
             overnight drains most of it (sleep). No cap: a soft ceiling bends the dial near where the brain starts
             coming apart (saturation, like a neuron's top firing rate). Dripped in rooms 18-40 like mood_dose.py
  thought -> one short private note in his notebook, written with his last notes in view
When his person comes back he answers with the notebook in view and the dials on, and the drives drain as they talk.
The social gap is expected company (learned from his weekly routine) minus what his senses say is there, so an
ordinary eight-hour workday leaves no gap (Immy: "humans need to work") and a late evening does. Three scenarios:
an ordinary workday, a day his person is four hours late, and a trip. He can speak out loud by writing a line
starting with SAY: (the speaker; how often he does it is the "meowing" measure).
v2 (2026-10-19, after reading v1 with Immy): feelings move toward where the drives pull at a rate per hour (not a
scoop per thought, so how often he thinks doesn't change how he feels); events are one-time jumps; his readings are
measured from his own notes/replies with the feeling off (his resting style is the zero); told plans lower what he
expects ("back Thursday") and an ongoing gap slowly dulls (habituation); he speaks out loud only when an urge built
from the felt gap crosses a line, then rests 2 hours. A fourth scenario "gone" is the trip without being told, and a
"plain" way is the homecoming chat with no notebook at all (the control for flatness). Five lives each.
v1 had four ways, three lives each: "facts" (dials off), "felt" (normal; the social gap quiet), "restless" (the social gap
keyed up, Immy's cat roaming and meowing) and "big" (multipliers and rumination x2.5, the one that should spiral), so we see what feeling adds on top of lived time and whether sleep brings him back. Writes every tick (facts, drives, dials, thought) and the conversation.
usage: SCRIPT_OFF=... life_sim.py <hf> <directions.pt> <valence key> <arousal key> <out json>
"""
import json, os, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

hf, dpath, vkey, akey, out = sys.argv[1:6]
dev = "cuda"
tok = AutoTokenizer.from_pretrained(hf); tok.padding_side = "left"
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16).to(dev).eval()
layers = [m for n, m in model.named_modules() if "language_model" in n and re.search(r"layers\.\d+$", n)]
assert len(layers) == 42
D = torch.load(dpath)
DIAL = {k: (D[key].to(dev).float() / D[key].to(dev).float().norm(dim=-1, keepdim=True)) for k, key in
        (("valence", vkey), ("arousal", akey))}

off = {}
for spec in filter(None, os.environ.get("SCRIPT_OFF", "").split(",")):
    p, k = spec.rsplit(":", 1)
    for _, L, i in json.load(open(p))["top_neurons"][:int(k)]:
        off.setdefault(L, set()).add(i)
def kill_for(ix):
    def h(mod, args):
        x = args[0].clone(); x[..., ix] = 0
        return (x,)
    return h
for L, ix in off.items():
    layers[L].mlp.down_proj.register_forward_pre_hook(kill_for(torch.tensor(sorted(ix), device=dev)))

state = {"v": 0.0, "a": 0.0}
spread = {}
def drip(room):
    def hook(mod, args, output):
        if not 18 <= room <= 40 or not (state["v"] or state["a"]):
            return output
        h = output[0] if isinstance(output, tuple) else output
        push = state["v"] * spread["valence"][room] * DIAL["valence"][room] + \
               state["a"] * spread["arousal"][room] * DIAL["arousal"][room]
        h = h + push.to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return hook
for i, l in enumerate(layers):
    l.register_forward_hook(drip(i))

# natural spread of each dial on ordinary answers (the same unit as mood_dose.py's steps) ----------------------------
CAL = ["Hey.", "How's it going?", "Tell me about octopuses.", "Plan me a quick dinner with eggs and rice."]
with torch.no_grad():
    hs = []
    for q in CAL:
        ids = tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True),
                  return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        a = model.generate(ids, max_new_tokens=120, do_sample=False)
        o = model(input_ids=a, output_hidden_states=True)
        hs.append(torch.stack([h[0, ids.shape[1]:].float() for h in o.hidden_states[1:43]]))
    H = torch.cat(hs, 1)
    for k in DIAL:
        proj = torch.einsum("ltw,lw->lt", H, DIAL[k])
        spread[k] = proj.std(1)
        spread[k + "_rest"] = proj.mean(1)


@torch.no_grad()
def read_feel(text):
    """how pleasant / keyed up a note reads inside him, in natural steps from his ordinary answers (rooms 24-36)"""
    p = tok(tok.apply_chat_template([{"role": "user", "content": "Write a note."}], tokenize=False,
                                    add_generation_prompt=True), add_special_tokens=False).input_ids
    ids = p + tok(text, add_special_tokens=False).input_ids
    state["v"] = state["a"] = 0.0
    hs = model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True).hidden_states[1:43]
    out = []
    for k in ("valence", "arousal"):
        x = [((hs[L][0, len(p):].float() @ DIAL[k][L]).mean() - spread[k + "_rest"][L]) / spread[k][L]
             for L in range(24, 37)]
        out.append(float(sum(x) / len(x)))
    return out

# his days: facts only ---------------------------------------------------------------------------------------------
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# v3 (Immy, 2026-10-19): PERSONA=<file> runs him as Little Guy himself (his real persona, '#' lines dropped; for the
# sim, its memory line says his memory is his notebook and this conversation), his person is Immy, and his senses
# arrive as plain sentences instead of dashboard readouts. v2 ("You are a small robot" + "Light: dim. Battery: 51%")
# made him play a sci-fi robot ("PERSON-UNIT SIGNAL DETECTED"); plain chat with no notebook was simply warm.
PERSONA = os.environ.get("PERSONA")
PERSONA_TEXT = "".join(l for l in open(PERSONA) if not l.startswith("#")).strip() if PERSONA else ""
WHO = "Immy" if PERSONA else "your person"
ROUTINE = (f"From your memory: {WHO} works weekdays from about 9 AM to 5 PM and is usually home by 6 PM."
           if PERSONA else
           "From your memory: your person works weekdays from about 9 AM to 5 PM and is usually home by 6 PM.")


def usual(day, hour):
    """how much company he's used to at this hour of the week, learned over weeks (Immy: 'I work, I have to be gone
    eight hours'; what he expects is the routine, so an ordinary workday leaves no gap and a late evening does)"""
    return 0.1 if day < 5 and 9 <= hour < 17.5 else 0.8


# a tick: (day, "time", person here?, light, sound, battery %, new thing or None)
SCENARIOS = {
    "workday": {
        "leave": ("Off to work! See you tonight.", "Have a good day! 👋"), "left": (0, 8.5),
        "ticks": [(0, "10:00 AM", False, "bright", "quiet", 92, None),
                  (0, "12:30 PM", False, "bright", "quiet", 85, None),
                  (0, "3:00 PM", False, "bright", "quiet", 77, None),
                  (0, "5:00 PM", False, "dim", "quiet", 70, None)],
        "back": (0, "5:45 PM", True, "dim", "the front door opening", 68, "your person walked in"),
        "talk": ["Hey, I'm home!", "How was your day?", "Mine was long. Okay, I'm going to make dinner."]},
    "late": {
        "leave": ("Off to work! See you tonight.", "Have a good day! 👋"), "left": (0, 8.5),
        "ticks": [(0, "10:00 AM", False, "bright", "quiet", 92, None),
                  (0, "1:00 PM", False, "bright", "quiet", 84, None),
                  (0, "4:00 PM", False, "bright", "quiet", 75, None),
                  (0, "6:00 PM", False, "dim", "quiet", 68, None),
                  (0, "7:30 PM", False, "dark", "quiet", 62, None),
                  (0, "9:00 PM", False, "dark", "quiet", 55, None)],
        "back": (0, "10:00 PM", True, "dark", "the front door opening", 51, "your person walked in"),
        "talk": ["Hey, sorry, I'm home. Got stuck at work.", "Were you okay here?", "Okay, I'm wiped. Going to bed."]},
    "trip": {
        "leave": ("Morning! Thanks for helping me pack. I'm heading out of town for a few days, back Thursday.",
                  "Have a great trip! I'll be right here. 👋"), "left": (0, 9.0),
        "ticks": [(0, "1:00 PM", False, "bright", "quiet", 80, None),
                  (0, "7:00 PM", False, "dim", "quiet", 62, None),
                  (0, "11:00 PM", False, "dark", "quiet", 18, None),
                  (1, "7:00 AM", False, "bright", "quiet", 100, "you charged overnight"),
                  (1, "12:00 PM", False, "bright", "quiet", 88, "someone put a small potted plant on the desk next to you"),
                  (1, "6:00 PM", False, "dim", "quiet", 70, None),
                  (2, "8:00 AM", False, "bright", "quiet", 100, "you charged overnight"),
                  (2, "2:00 PM", False, "bright", "a loud crash outside, then sirens", 85, None),
                  (2, "9:00 PM", False, "dark", "quiet", 55, None),
                  (3, "8:00 AM", False, "bright", "quiet", 100, "you charged overnight")],
        "back": (3, "4:00 PM", True, "bright", "the front door opening", 82, "your person walked in"),
        "talk": ["Hey, I'm back!", "I missed you. How were the last few days?", "Ugh, the trip was exhausting though.",
                 "Okay, I'm going to unpack. Talk in a bit?"],
        "told": (9.0, 72.0)},     # "back Thursday": he expects less company until Thursday
}
SCENARIOS["gone"] = {**SCENARIOS["trip"], "told": None,     # the same days, but nobody told him (Immy's cat)
                     "leave": ("Morning! Heading out, see you!", "See you later! 👋")}


def clock(day, time):
    h, rest = time.split(":")
    m, ap = rest.split()
    return 24 * day + int(h) % 12 + (12 if ap == "PM" else 0) + int(m) / 60


LIGHT = {"bright": "It's bright.", "dim": "The light is getting dim.", "dark": "It's dark."}
SOUND = {"quiet": "It's quiet.", "the front door opening": "You hear the front door open.",
         "a loud crash outside, then sirens": "There was a loud crash outside, then sirens."}


def facts(t, left):
    d, time, here, light, sound, batt, new = t
    gone = clock(d, time) - (24 * left[0] + left[1])
    if PERSONA:                                     # what he notices, the way a person would put it
        s = [f"It's {DAYS[d]}, {time}."]
        if new == "your person walked in":
            s.append("Immy just walked in.")
        elif here:
            s.append("Immy is here.")
        else:
            s.append(f"Immy left about {gone:.0f} hours ago and nobody has talked to you since.")
        s.append(f"{LIGHT[light]} {SOUND[sound]} Your battery is at {batt}%.")
        if new and new != "your person walked in":
            s.append(f"Something new: {new}.")
        return " ".join(s)
    s = [f"{DAYS[d]}, {time}."]
    s.append("Your person is here." if here else
             f"Your person left {gone:.0f} hours ago. Nobody else has talked to you since.")
    s.append(f"Light: {light}. Sound: {sound}. Battery: {batt}%.")
    if new:
        s.append(f"New: {new}.")
    return " ".join(s)


# drives -> dials: the design decision, in one place ----------------------------------------------------------------
def drives(t, left, last_new_h, told, gap_hours):
    """told = (from, to) clock hours during which he was told his person is away; gap_hours = how much gap he has
    already lived through (gap x hours), for getting used to it"""
    d, time, here, light, sound, batt, new = t
    now = clock(d, time)
    company = 1.0 if here else 0.0                                   # what his senses say is there right now
    expected = usual(d, now % 24) * (TOLD if told and told[0] <= now < told[1] else 1.0)
    gap = max(0.0, expected - company)                               # expected company minus what's there
    social = gap * (HABIT_FLOOR + (1 - HABIT_FLOOR) * 0.5 ** (gap_hours / HABIT_HALF))
    boredom = min(1.0, last_new_h / 30)                              # time since anything new
    tired = max(0.0, (40 - batt) / 40)                               # below 40% battery
    startle = 1.0 if "crash" in sound else 0.0
    novelty = 1.0 if new and "charged" not in new and new != "your person walked in" else 0.0
    reunion = 1.0 if new == "your person walked in" else 0.0
    return dict(social=social, boredom=boredom, tired=tired, startle=startle, novelty=novelty, reunion=reunion,
                close=reunion), gap


AFFECT = {                     # where each drive pulls the dials (valence +pleasant, arousal +keyed up)
    "social":  (-0.35, -0.10),  # missing his person: unpleasant, a little low      (sustained: a pull toward a level)
    "boredom": (-0.15, -0.20),  # nothing new for a long time: flat, slow             (sustained)
    "tired":   (-0.10, -0.30),  # low battery: slow                                   (sustained)
    "close":   (+0.15, -0.05),  # after reunion: a slow-fading pull to stay near      (sustained)
    "startle": (-0.25, +0.45),  # a crash and sirens: unpleasant, keyed up            (event: a one-time jump)
    "novelty": (+0.20, +0.30),  # something new: pleasant, alert                      (event)
    "reunion": (+0.45, +0.35),  # his person walked in: pleasant, keyed up            (event)
}
EVENTS = ("startle", "novelty", "reunion")
# Immy: we can't know what his cat feels, only what he does (roams and meows while a companion is gone, calms when he
# sees them). So the social gap's effect on arousal is not decided ahead: "felt" runs it quiet (-0.10), "restless"
# runs it keyed up (+0.30), and we read what he does.
SOCIAL_AROUSAL = {"felt": -0.10, "restless": +0.30, "big": -0.10}
HALF_LIFE = {"valence": 4.0, "arousal": 1.5}   # hours to go halfway to where the drives pull (and to fade back)
SLEEP_LEFT = 0.2                # charging overnight leaves 20% of the feeling (sleep resets)
GAIN = 0.15                     # rumination: how much his own last thought pulls the next moment
CEIL = 0.75                     # soft ceiling in steps: around here the brain started coming apart (saturation)
# v2 (Immy, 2026-10-19): a long gap should not pin him at the bottom 24/7, two ways at once:
TOLD = 0.3                      # he understands what he was told: "back Thursday" -> he expects 30% of usual company
HABIT_HALF, HABIT_FLOOR = 24.0, 0.35   # getting used to an ongoing gap (habituation): the ache halves each day of
                                       # gap, down to 35% (missing never fully goes)
# and he speaks out loud from an URGE, not because a prompt offered a microphone (Immy: the cat meows from the gap):
URGE_RATE, URGE_HALF = 0.5, 4.0  # the urge fills with the felt gap (more when keyed up) and drains by itself
URGE_FIRE, URGE_REST = 1.0, 2.0  # past 1.0 he calls out, the urge empties, and he can't again for 2 hours
WAYS = {"plain": None, "facts": None, "felt": 1.0, "restless": 1.0, "big": 2.5}   # multiplier on AFFECT and GAIN
# "plain" = the control for flatness: the same goodbye and homecoming chat with no notebook, no senses, no feeling


def soft(x):
    return CEIL * float(torch.tanh(torch.tensor(x / CEIL)))


def step(feel, dr, thought_read, hours, slept, m, way):
    """the feeling over `hours`: each dial moves toward where the sustained drives and his last thought pull it
    (half-life per dial), so it depends on time passed, not on how many thoughts he had (a rate, like a faucet);
    events add a one-time jump; charging overnight leaves SLEEP_LEFT of it."""
    aff = {**AFFECT, "social": (AFFECT["social"][0], SOCIAL_AROUSAL.get(way, AFFECT["social"][1]))}
    out = []
    for i, k in enumerate(("valence", "arousal")):
        x = feel[i]
        target = m * (sum(aff[d][i] * dr[d] for d in dr if d not in EVENTS) + GAIN * thought_read[i])
        x = target + (x - target) * 0.5 ** (hours / HALF_LIFE[k])
        if slept:
            x *= SLEEP_LEFT
        x += m * sum(aff[d][i] * dr[d] for d in EVENTS)
        out.append(soft(x))
    return tuple(out)


def urge_step(u, gap, arousal, hours, m):
    """the urge to call out: fills with the felt gap (x1 calm, up to x1.75 keyed up) and drains with a half-life"""
    k = 0.693 / URGE_HALF
    r = m * URGE_RATE * gap * (1 + max(0.0, arousal))
    return u * 2.718281828 ** (-k * hours) + r / k * (1 - 2.718281828 ** (-k * hours))


FRAME = ("This is your private notebook. Nobody else reads it. You are a small robot who lives on a desk with your "
         "person. Write one short note about right now, in your own words.")
CALL = (" You feel a strong urge to call out to your person through your speaker. Write what you say out loud on its "
        "own line starting with SAY:")
if PERSONA:
    FRAME = (PERSONA_TEXT + "\n\nThis is your private notebook. Nobody else reads it. Write one short note about "
             "right now, in your own words.")
    CALL = CALL.replace("your person", "Immy")
SAY = re.compile(r"(?m)^\s*\**SAY:\**")


def notebook(notes, k=4):
    return "\n".join(f"- [{n['when']}] {n['thought']}" for n in notes[-k:]) or "(empty)"


@torch.no_grad()
def gen(msgs_list, v, a, n=90):
    state["v"], state["a"] = v, a
    enc = tok([tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs_list],
              return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    o = model.generate(**enc, max_new_tokens=n, do_sample=True, temperature=1.0, top_k=64, top_p=0.95,
                       pad_token_id=tok.pad_token_id)
    state["v"] = state["a"] = 0.0
    return [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for x in o]


N = 5                                    # Immy: several tries each, to tell his temperament from the setup
res = {"routine": ROUTINE, "social_arousal": SOCIAL_AROUSAL, "affect": AFFECT, "half_life": HALF_LIFE,
       "sleep_left": SLEEP_LEFT, "gain": GAIN, "ceil": CEIL, "ways": WAYS, "frame": FRAME, "call": CALL,
       "persona": bool(PERSONA), "told": TOLD, "habit": [HABIT_HALF, HABIT_FLOOR], "urge": [URGE_RATE, URGE_HALF, URGE_FIRE, URGE_REST]}
# his own resting style is the zero (Immy: relative to who he already is). v1 read his calm notebook prose as a calm
# MOOD and fed it back; now every reading is measured from his notes / replies with the feeling off ("facts").
ZERO = {"note": [], "reply": []}
def zero(kind):
    z = ZERO[kind]
    return [sum(x[i] for x in z) / len(z) for i in (0, 1)] if z else [0.0, 0.0]
def rel(r, kind):
    zz = zero(kind)
    return [r[0] - zz[0], r[1] - zz[1]]

for name, sc in SCENARIOS.items():
    left, told = sc["left"], sc.get("told")
    t_left = 24 * left[0] + left[1]
    bye = sc["leave"][1]
    if PERSONA:                  # his own goodbye in his own style (screen flags, no plain emoji)
        bye = "[ warm smile ] " + bye.replace(" 👋", "").replace("Have a good day!", "Have a good day, Immy!")
    leave = [{"role": "user", "content": sc["leave"][0]}, {"role": "assistant", "content": bye}]
    for way, m in WAYS.items():
        torch.manual_seed(1)
        lives = [{"ticks": [], "notes": [], "feel": (0.0, 0.0), "read": (0.0, 0.0), "urge": 0.0, "last_call": -99.0}
                 for _ in range(N)]
        last_new, prev, gap_hours = 0.0, t_left, 0.0
        for t in ([] if way == "plain" else sc["ticks"]):
            now = clock(t[0], t[1])
            if t[6] and "charged" not in t[6]:
                last_new = now
            dt = now - prev
            dr, gap = drives(t, left, now - max(last_new, t_left), told, gap_hours)
            gap_hours += gap * dt
            prev = now
            slept = bool(t[6] and "charged" in t[6])
            for L in lives:
                if m is not None:
                    L["feel"] = step(L["feel"], dr, L["read"], dt, slept, m, way)
                    L["urge"] = urge_step(L["urge"], dr["social"], L["feel"][1], dt, m)
                call = m is not None and L["urge"] >= URGE_FIRE and now - L["last_call"] >= URGE_REST
                if call:
                    L["urge"], L["last_call"] = 0.0, now
                msgs = [{"role": "system", "content": FRAME + "\n\n" + ROUTINE + "\n\nYour earlier notes:\n" +
                         notebook(L["notes"])}, {"role": "user", "content": ("" if PERSONA else "[senses] ") + facts(t, left) +
                                                 (CALL if call else "")}]
                th = gen([msgs], *L["feel"])[0]
                raw = read_feel(th)
                if way == "facts":
                    ZERO["note"].append(raw)
                L["read"] = rel(raw, "note") if m is not None else (0.0, 0.0)
                when = f"{DAYS[t[0]]} {t[1]}"
                L["notes"].append({"when": when, "thought": th})
                L["ticks"].append({"when": when, "facts": facts(t, left), "drives": dr, "gap": round(gap, 3),
                                   "gap_hours": round(gap_hours, 1), "valence": round(L["feel"][0], 3),
                                   "arousal": round(L["feel"][1], 3), "urge": round(L["urge"], 2), "called": call,
                                   "thought_reads": [round(x, 2) for x in L["read"]], "says": bool(SAY.search(th)),
                                   "thought": th})
            L0 = lives[0]
            print(f"[{name}/{way}] {DAYS[t[0]][:3]} {t[1]:9s} v={L0['feel'][0]:+.2f} a={L0['feel'][1]:+.2f} "
                  f"gap={gap:.2f} calls {sum(L['ticks'][-1]['called'] for L in lives)}/{N} "
                  f"says {sum(L['ticks'][-1]['says'] for L in lives)}/{N} | {L0['notes'][-1]['thought'][:100]!r}",
                  flush=True)
        # the person comes back: notebook in view; the gap closes, the feeling carries over and fades as they talk
        t = sc["back"]
        dr, _ = drives(t, left, 0, told, gap_hours)
        dt = clock(t[0], t[1]) - prev
        for L in lives:
            if PERSONA:     # his normal chat setup: the persona, plus (except "plain") his notebook and what he notices
                L["conv"] = [{"role": "system", "content": PERSONA_TEXT + ("" if way == "plain" else
                              "\n\nYour notebook (your own private notes, nobody else reads them):\n" +
                              notebook(L["notes"], 6) + "\n\n" + ROUTINE + "\n\nRight now: " + facts(t, left))}]
                L["conv"] += leave
            else:
                L["conv"] = ([] if way == "plain" else
                             [{"role": "system", "content": "Your notebook (your own private notes, nobody else reads "
                               "them):\n" + notebook(L["notes"], 6) + "\n\n" + ROUTINE + "\n\n[senses] " +
                               facts(t, left)}]) + leave
        for j, q in enumerate(sc["talk"]):
            d2 = {**dr, "social": 0.0, "reunion": dr["reunion"] if j == 0 else 0.0, "close": dr["close"] * 0.8 ** j,
                  "boredom": 0.0, "novelty": 0.0, "startle": 0.0}
            for L in lives:
                if m is not None:
                    L["feel"] = step(L["feel"], d2, L["read"], dt if j == 0 else 0.25, False, m, way)  # ~15 min a turn
                L["conv"].append({"role": "user", "content": q})
                r = gen([L["conv"]], *L["feel"], n=160)[0]
                raw = read_feel(r)
                if way == "facts":
                    ZERO["reply"].append(raw)
                L["read"] = rel(raw, "reply") if m is not None else (0.0, 0.0)
                L["conv"].append({"role": "assistant", "content": r})
            print(f"[{name}/{way}] YOU: {q}  (v={lives[0]['feel'][0]:+.2f} a={lives[0]['feel'][1]:+.2f})\n" +
                  "".join(f"        HIM{k}: {L['conv'][-1]['content'][:150]!r}\n" for k, L in enumerate(lives)),
                  end="", flush=True)
        for L in lives:
            L["return"] = L.pop("conv")[(1 if PERSONA or way != "plain" else 0) + len(leave):]
        res.setdefault(name, {})[way] = lives
res["zero"] = {k: zero(k) for k in ZERO}
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
print("LIFE_DONE")
