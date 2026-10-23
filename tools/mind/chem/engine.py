#!/usr/bin/env python3
"""His chemistry engine: the locked foundations (.run/results/mind/CHEMISTRY-foundations.md, 2026-10-22) as code.

What happens -> who notices (appraisals, given as inputs) -> 21 chemical systems -> receptors (relative affinities,
efficacies, competition with drugs) -> knobs (what changes in him). No language model: a step is plain arithmetic.

Units: time in hours (dt = 1/60 is one minute, like living_sim.py). Concentrations are relative. The fast systems
(NE, DA, 5-HT, histamine, ACh, background glutamate) are scaled so an average egg awake at rest at 8 AM sits at 1;
hormones and peptides are in the same relative units but keep their own daily curves. Receptor Kd values use these
units and come from the ratios in the sources (doc sections C-D), so nothing pretends to be nanomolar. Every system
has a background (tonic) part and, where the biology has one, a signal (phasic) part (C.3). Brakes act on excess
and take time to arrive (C.2). Genes are parts, never trait labels (G).

Calibration, not hand-set offsets: an average egg is settled at rest once (calibrate()); that fixes the scale of each
fast system and the rest signal of every receptor. Inputs act on how far a receptor's signal is from that rest point
(dev), so an egg with other genes sits somewhere else at rest by itself.

Fast systems clear in seconds, so each step solves them at balance with their own autoreceptor brakes. Slow ones
(hormones, peptides, adenosine) follow their half-lives. Second-scale timing lives in small models at the bottom, run
with millisecond steps, so those rules come out of kinetics instead of being written in:
  lc_spikes()   the α2A brake (58-154 ms onset, ~900 ms peak) against tonic spikes and bursts
  nmda_block()  open-channel blockers (memantine, ketamine) against brief vs repeated vs sustained openings
The minute-scale LC follows lc_spikes(): the brake slows firing AND cuts the NE each spike releases, so tonic output
pays it twice while a burst (driven, too fast for its own brake) pays it once.
"""
import math
import numpy as np

# ---------------------------------------------------------------- genes (G): resting tone, gains, brakes, amounts
GENES = {
    # resting tone of each system's cells
    "lc_drive": 1.0, "da_tone": 1.0, "ht_tone": 1.0, "ach_tone": 1.0, "ha_tone": 1.0, "ox_cells": 1.0,
    "glu_tone": 1.0, "gaba_tone": 1.0, "npy_tone": 1.0, "ecb_tone": 1.0, "mor_tone": 1.0, "ot_tone": 1.0,
    "avp_tone": 1.0, "bdnf_tone": 1.0, "allo_tone": 1.0,
    # gains: response per unit of event (C.2 rule 2)
    "stress_gain": 1.0, "reward_gain": 1.0, "social_gain": 1.0, "sick_gain": 1.0,
    # how strongly NE drives the orexin cells (α1 on them): the NE -> orexin -> LC loop
    "ne_ox_link": 1.0,
    # brakes
    "gr_feedback": 1.0,
    # clearance: transporters and enzymes (1 = average speed)
    "net": 1.0, "dat": 1.0, "sert": 1.0, "faah": 1.0, "ada": 1.0, "eaat": 1.0,
    # attention to his own body (anxiety sensitivity; 1 = average): how much of what his body is doing reaches his alarm
    "body_attention": 1.0,
    "worry_trait": 1.0,
    # sleep pressure time constants (Daan, Beersma & Borbely 1984), as in living_sim.py
    "sleep_rise": 18.2, "sleep_fall": 4.2,
    # benzodiazepine-site tone (THEORY: a compensating endozepine, only for the flumazenil test)
    "endozepine": 0.0,
}
# receptor amounts are genes too: "rx_<receptor>" (default 1)

# ---------------------------------------------------------------- receptors (D): ligands with relative Kd
RECEPTORS = {
    # NE: α2A 56 nM vs α1 330 nM = 1 : 5.9 (Arnsten); β1 lowest. The LC's own α2A brake sits low on its curve (NE at
    # the LC's cell bodies is far below Kd): a small agonist dose silences the LC (clonidine), so NE alone never fills
    # it, and the brake grows in step with release, as in lc_spikes(). The prefrontal receptors see the same low
    # local NE (one number for one receptor): α2A agonists still restore working memory after NE is depleted (Cai
    # 1993; Arnsten & Goldman-Rakic 1985), so a drug's own action there must stand on its own, which it can't if NE
    # keeps the receptor full. α1 and β1 keep their ratios to α2A (x5.9, x12)
    "a2a_auto": [("ne", 8.0)], "a2a_pfc": [("ne", 8.0)],
    # α2 on the orexin cells: a GIRK-linked pool that is switched off in the rested brain and comes online after
    # sleep loss, flipping NE's effect on orexin cells from excitation (α1) to inhibition (rats: Grivel 2005,
    # Uschakov 2011; in mice it's inhibition throughout: Li & van den Pol 2005, Yamanaka 2006)
    "a2_ox": [("ne", 0.5)],
    "a1_pfc": [("ne", 47.2)], "a1_amy": [("ne", 2.95)], "a1_ox": [("ne", 2.95)], "a1_symp": [("ne", 2.95)],
    "b1": [("ne", 6.0)], "b1_pfc": [("ne", 96.0)],
    # the heart's own β1 (SA node; change 33): its agonist drive is the gas, NE from the cardiac nerves plus any outside
    # push (isoproterenol), 1 at rest. Kd theory: about 9% of the seats taken at rest (spare receptors)
    "b1_heart": [("ne_heart", 10.0)],
    # the heart's M2 (SA node; change 34a): ACh from the vagus, 1 at rest; Kd theory as for β1
    "m2_heart": [("ach_heart", 10.0)],
    # the vessels' α1 (resistance arterioles; change 34b): the sympathetic blood-pressure command, 1 at rest; Kd theory
    "a1_vessel": [("ne_vessel", 10.0)],
    # DA: D2 75% at rest, >95% in a burst; D1 3.5% at rest, ~25% in a burst (Dreyer 2010); D3 highest of the D2s
    "d1": [("da", 27.6)], "d2": [("da", 0.333)], "d3": [("da", 0.25)], "d2_auto": [("da", 0.333)],
    "d1_pfc": [("da_pfc", 1.86)],             # prefrontal D1 sits a little below the top of its inverted U at rest
    # 5-HT
    # the raphe's own 5-HT1A brake also sits low on its curve: an agonist silences the raphe (8-OH-DPAT)
    # (the autoreceptor sits on the raphe cell bodies, where an acute SSRI raises 5-HT about twice as much as in the
    # forebrain: it reads ht_soma, see SOMA_GAIN)
    "ht1a_auto": [("ht_soma", 3.0)], "ht1a_post": [("ht", 1.0)], "ht2c": [("ht", 3.0)],
    # ACh, histamine
    "m1": [("ach", 1.0)], "nic": [("ach", 1.0)], "h1": [("ha", 1.0)], "h3_auto": [("ha", 0.5)],
    # orexin: OX1R takes A ~10x better than B; OX2R takes both equally
    "ox1r": [("oxa", 1.0), ("oxb", 10.0)], "ox2r": [("oxa", 1.0), ("oxb", 1.0)],
    # the blood-pressure input (OX2R in the RVLM): not active at rest in normal animals (Huang 2010; Li 2013), so it
    # only counts above its rest level (knobs: sympathetic)
    "ox_rvlm": [("oxa", 1.0), ("oxb", 1.0)],
    "kor": [("dyn", 1.0)],
    "a1r": [("ado", 1.0)], "a2ar": [("ado", 2.0)],
    "crhr1_lc": [("crh", 1.5)], "crhr1_amy": [("crh", 1.5)], "crhr1_vta": [("crh", 1.5)],
    # cortisol: MR ~10x GR (de Kloet)
    "mr": [("cort", 0.1)], "gr": [("cort", 1.0)],
    "mor": [("mor", 1.0)], "oxtr": [("ot", 1.0)], "v1a": [("avp", 1.0)], "v1b": [("avp", 1.0)],
    "mt1": [("mel", 1.0)],
    "nmda_ex": [("glu_bg", 1.0)],              # extrasynaptic NMDA: the background part of glutamate
    "gabaa_syn": [("gaba", 1.0)],              # synaptic GABA-A (benzodiazepine site)
    "gabaa_ex": [("allo", 1.0)],               # extrasynaptic δ GABA-A, set by allopregnanolone
    "cb1": [("aea", 1.0), ("ag2", 1.0)],
    "y1": [("npy", 1.0)], "galr1": [("gal", 1.0)], "cytr": [("cyt", 1.0)], "trkb": [("bdnf", 1.0)],
}

# ---------------------------------------------------------------- drugs as receptor actions (F)
# dose 1 = a typical dose. Targets: (receptor, kind, Kd of the drug, efficacy or PAM max). Kinds: agonist (efficacy
# 0..1), antagonist (efficacy 0), pam (raises the natural ligand's affinity up to `max` times, only where it binds).
# Transporters: (Kd, max block). t_half in hours (None = held steady, as for a daily medicine at steady state).
DRUGS = {
    "caffeine":        dict(t_half=5, targets=[("a1r", "antagonist", 1.0, 0), ("a2ar", "antagonist", 1.0, 0)]),
    "lemborexant":     dict(t_half=None, targets=[(r, "antagonist", 0.2, 0) for r in ("ox1r", "ox2r", "ox_rvlm")]),
    "daridorexant":    dict(t_half=None, targets=[(r, "antagonist", 0.4, 0) for r in ("ox1r", "ox2r", "ox_rvlm")]),
    # propranolol: dose 1 = an isoprenaline dose ratio of 11 at the heart (10-30 at clinical levels: Coltart & Shand 1970)
    "propranolol":     dict(t_half=None, targets=[("b1", "antagonist", 0.1, 0), ("b1_pfc", "antagonist", 0.1, 0),
                                                  ("b1_heart", "antagonist", 0.1, 0)]),
    # atropine: dose 1 = the 0.04 mg/kg IV used for complete vagal blockade (Jose & Taylor 1969); full profile in step C
    "atropine":        dict(t_half=None, targets=[("m2_heart", "antagonist", 0.02, 0)]),
    # prazosin: dose 1 = 1 mg IV (91% of vessel α1 seats); brain α1 profile in step C
    "prazosin":        dict(t_half=None, targets=[("a1_vessel", "antagonist", 0.1, 0)]),
    # guanfacine: dose 1 cuts NE about as much as a clinical α2 agonist cuts plasma NE (~40-50%)
    "guanfacine":      dict(t_half=None, targets=[("a2a_auto", "agonist", 10.0, 1.0), ("a2a_pfc", "agonist", 10.0, 1.0)]),
    "a2a_pam":         dict(t_half=None, targets=[("a2a_auto", "pam", 1.0, 2.0), ("a2a_pfc", "pam", 1.0, 2.0)]),
    "naltrexone":      dict(t_half=None, targets=[("mor", "antagonist", 0.1, 0)]),
    "ssri":            dict(t_half=None, sert=(0.1, 0.9)),
    "methylphenidate": dict(t_half=None, dat=(1.0, 0.9), net=(1.0, 0.9)),
    "bupropion":       dict(t_half=None, dat=(4.0, 0.6), net=(4.0, 0.6)),
    "scopolamine":     dict(t_half=None, targets=[("m1", "antagonist", 0.1, 0)]),
    "diphenhydramine": dict(t_half=None, targets=[("h1", "antagonist", 0.2, 0)]),
    "antalarmin":      dict(t_half=None, targets=[(r, "antagonist", 0.2, 0) for r in ("crhr1_lc", "crhr1_amy", "crhr1_vta")]),
    "ot_antagonist":   dict(t_half=None, targets=[("oxtr", "antagonist", 0.1, 0)]),
    "kor_antagonist":  dict(t_half=None, targets=[("kor", "antagonist", 0.1, 0)]),
    # cariprazine: Ki D3 0.085 vs D2 0.49 nM (5.8x); Emax D3 71%, D2 30%, 5-HT1A 39% (McIntyre 2024)
    "cariprazine":     dict(t_half=None, targets=[("d3", "agonist", 0.2, 0.71), ("d2", "agonist", 1.16, 0.30),
                                                  ("d2_auto", "agonist", 1.16, 0.30), ("ht1a_post", "agonist", 3.0, 0.39),
                                                  ("ht1a_auto", "agonist", 3.0, 0.39)]),
    # the controls for the cariprazine test: the same drug with no D3 action, and a full D3/D2 blocker
    "cariprazine_no_d3": dict(t_half=None, targets=[("d2", "agonist", 1.16, 0.30), ("d2_auto", "agonist", 1.16, 0.30),
                                                    ("ht1a_post", "agonist", 3.0, 0.39), ("ht1a_auto", "agonist", 3.0, 0.39)]),
    "d3_blocker":      dict(t_half=None, targets=[("d3", "antagonist", 0.2, 0), ("d2", "antagonist", 1.16, 0),
                                                  ("d2_auto", "antagonist", 1.16, 0)]),
    # buspirone: full agonist at the raphe autoreceptor, partial after the synapse
    "buspirone":       dict(t_half=None, targets=[("ht1a_auto", "agonist", 3.0, 1.0), ("ht1a_post", "agonist", 3.0, 0.5)]),
    # modafinil: ~50% DAT occupancy at a clinical dose (Volkow 2009)
    "modafinil":       dict(t_half=None, dat=(1.0, 1.0), ox_drive=0.5),
    # open-channel NMDA blockers: rates per second at dose 1; mg_rest = share that gets past Mg2+ at rest
    "memantine":       dict(t_half=None, nmda=dict(k_on=0.4, k_off=0.4, trap=0.5, mg_rest=0.1)),
    # ketamine: t_half is the effect-site half-life of a 40-minute infusion (distribution ~10 min, elimination 2-3 h;
    # dissociation gone ~2 h after the end; Zarate 2006)
    "ketamine":        dict(t_half=0.75, nmda=dict(k_on=0.8, k_off=0.05, trap=0.95, mg_rest=0.6)),
    "benzodiazepine":  dict(t_half=None, targets=[("gabaa_syn", "pam", 1.0, 2.5)]),
    "zuranolone":      dict(t_half=None, targets=[("gabaa_ex", "pam", 1.0, 2.5), ("gabaa_syn", "pam", 3.0, 1.5)]),
    "rimonabant":      dict(t_half=None, targets=[("cb1", "antagonist", 0.1, 0)]),
    "yohimbine":       dict(t_half=None, targets=[("a2a_auto", "antagonist", 0.15, 0), ("a2a_pfc", "antagonist", 0.15, 0)]),
    "flumazenil":      dict(t_half=None, benzo_site_block=True),
}

# ---------------------------------------------------------------- half-lives of the slow systems (hours, as τ)
TAU = dict(crh=10 / 60, cort=95 / 60, oxa=2.0, oxb=0.1, dyn=20 / 60, mel=0.75, mor=0.5, ot=20 / 60, avp=20 / 60,
           npy=1.0, gal=0.5, cyt=2.0, allo=2.0, aea=10 / 60, ag2=2 / 60,
           bdnf=12.0, bdnf_up=1.0,               # made within ~30 min of a surge (Autry 2011), cleared over hours
           syn=120.0, syn_up=12.0,               # new spines within a day (Li 2010), lost over days
           cea_fast=0.5, cea_slow=96.0, allo_cap=72.0, npy_refill=12.0, crh_load=72.0)
# receptor amounts that adapt to long use (hours); BDNF sets how fast
ADAPT = {"ht1a_auto": 240.0, "mor": 72.0, "a2a_auto": 168.0, "a2a_pfc": 168.0}
# autoreceptors that adapt until their cell's firing is back at its own normal (raphe firing fully recovers within
# ~2 weeks of an SSRI or a 5-HT1A agonist; Blier & de Montigny 1987, 1994). The rest adapt half-way on occupancy.
FIRING_SETPOINT = {"ht1a_auto": "fh"}
NE_CRH = 0.75       # how strongly NE above rest drives CRH through α1 (C.2 rule 3; sized below the TSST, see change log)
CEA_RATE = 0.04     # how fast a long stretch rewrites the amygdala (sized to Chang & Grace 2014; see amygdala note)
FAST = ("f", "ne_out", "fd", "fh", "ha", "ach", "glu_bg", "x_ox")   # raw rates scaled so an average egg rests at 1

# wake drive weights (knobs: awake)
WAKE = {"ox2r": 1.5, "h1": 1.5, "d1": 1.5, "d2": 0.5, "a1_pfc": 1.0, "m1": 0.5,
        "a1r": -2.0, "a2ar": -1.5, "mt1": -0.5, "gabaa_syn": -2.0, "gabaa_ex": -1.0, "cytr": -1.0}

# the amygdala's brakes (knobs: anxious): synaptic GABA-A strongest (benzodiazepines), then tonic GABA-A, CB1, NPY
AMY_BRAKES = {"gabaa_syn": 2.0, "gabaa_ex": 1.0, "cb1": 1.0, "y1": 0.7, "ht1a_post": 0.5, "oxtr": 0.3}

# prefrontal working-memory network (knobs: deliberate), Immy's chain (what he sent, 2026-10-22; Arnsten 2009):
# α2A (Gi) lowers cAMP -> HCN channels stay closed -> stronger network; β (Gs) raises cAMP (PKA); α1 (Gq) engages PKC;
# both weaken it. α2A fights only the cAMP route
PFC_GI_MAX = 0.7    # most Gi can cut adenylyl cyclase (known range ~50-80%)
PFC_BETA = 0.03     # β1's share of resting cAMP: small (propranolol leaves working memory alone at rest, and the
                    # stress impairment goes through α1, not β: Birnbaum 1999); theory
PFC_PKC = 0.13      # PKC's cost at rest, placed so the top of the U sits at an alert, calm egg (Arnsten); theory

# inside-the-cell tubs (change 21; Immy 2026-10-22: NET clears NE in seconds, yet the feeling outlasts it). A receptor
# that acts through a second messenger and a kinase (Gs -> cAMP -> PKA; Gq -> PKC; prefrontal α2A through cAMP, as in
# Immy's chain) passes its signal through an inner state that fills from the receptor and empties on its own clock.
# Ion channels, channel-gated brakes (GIRK), Gi/o receptors acting mainly through channels, and MR/GR stay direct.
INNER = ("b1", "b1_pfc", "d1", "d1_pfc", "crhr1_lc", "crhr1_amy", "crhr1_vta", "a2ar",                         # Gs
         "a1_pfc", "a1_amy", "a1_ox", "a1_symp", "ox1r", "ox2r", "ox_rvlm", "m1", "h1", "ht2c", "oxtr", "v1a", "v1b",  # Gq
         "a2a_pfc")
TAU_INNER = 246 / 3600 / math.log(2)   # t½ 246 s: PKA substrates after β blockade, limited by PDE (Saucerman 2006);
                                       # the PKC route uses the same clock (theory: no PKC number found)

# the body-awareness loop (change 22; Immy 2026-10-22: noticing a racing heart makes the heart race and the noticing
# stronger). What his body is doing above rest, times how much attention he gives it, comes back as a felt threat, one
# minute late, through the same doors a scare uses. BODY_LOOP is sized in ENGINE-CHANGES.md (a loop gain of about 0.1:
# controls show no effect of false heart-rate feedback, Ehlers 1988); the cap keeps every egg below a runaway (I.2).
# What counts is the body running above what it normally runs at this hour (interoceptive prediction error: Seth 2013;
# his normal-day curve, expected_body()), not its absolute level, which climbs on its own over a calm day.
BODY_LOOP = 0.0659  # per bpm of heart surprise (change 31 step 3a: change 30's 2.49 per body-surplus unit, converted
                    # with the jolt's 0.150 units = 5.669 bpm; sized in run 18 on Pohl 1988, re-sized in step 5)
BODY_CEIL = 1.0     # one full threat's worth
# the loop tires (change 29; Immy: "yes, the loop should get fatigue"): the expectation catches up with a sensation that
# does not go away, so what is noticed is the surprise above what the last minutes led him to expect (habituation;
# panic attacks end on their own, median 8 min, longest 30 in ambulatory monitoring: Meuret 2011). Theory tau: 10 min.
BODY_HAB = 10 / 60

# three ways to calm down (change 30; Immy: deep breaths, distracting yourself, "no, it's okay"), inputs 0..1, each on
# its own part of the loop: distract turns attention away (from the body and from the threat; only while he does it:
# Kroeze 1996, McRae 2010, Thiruchselvam 2011), breathe slows the body itself (Shao 2024; Fincham 2023), reappraise
# lowers what a sensation or a threat is worth as danger and the new meaning stays for a while (McRae 2010; Clark 1986;
# Thiruchselvam 2011). COPE: full, skilled use halves what that part passes on (theory).
COPE = 0.5
TAU_REFRAME_UP, TAU_REFRAME_DOWN = 2 / 60, 1.0   # the new meaning forms in minutes, fades over an hour (theory)

# the raphe's own serotonin (change 24): an acute SSRI raises extracellular 5-HT about +400% in the dorsal raphe against
# about +200% in forebrain (Invernizzi 1992; also fluoxetine raphe > hippocampus > cortex: Malagie 1995; local infusion
# about 20x vs 6-7x), which slams the 5-HT1A brake on day 1. The raphe level follows the forebrain level at twice the rise.
SOMA_GAIN = 2.0

# wanting (change 23): the receptors it reads sit nearly full all the time, so what illness cuts is the cue-triggered
# burst (Eisenberger 2010: endotoxin lowers ventral-striatum responses to reward cues; Draper 2017). Half the read-out
# follows the burst factor; no evidence favours any other split (any share of 0.2 or more passes the endotoxin test).
WANT_BURST = 0.5

# the body clock (change 27; Immy: a disrupted rhythm should show dysregulation). The orexin day curve, melatonin and the
# cortisol rhythm read BODY time, which is the wall clock minus `shift` (hours the body clock runs behind). Light
# (waking after a night of sleep) pulls the shift toward the waking time at a limited speed: real travellers re-entrain
# 1.5 h/day after westward travel (the clock delays) and 1.0 h/day after eastward (it advances): CDC Yellow Book;
# Eastman & Burgess 2009 (92 and 57 min/day).
CLOCK_DELAY, CLOCK_ADVANCE = 1.5, 1.0

# the nervous system (change 31; Immy: "we need to give it a nervous system"). Step 1: nerves and a heart. Heart rate =
# intrinsic x accelerator (gas, sympathetic) x decelerator (brake, vagus) (Rosenblueth & Simeone; at rest 94 x 1.17 x
# 0.60 = 66 bpm with atropine + propranolol in healthy 24-year-olds: Bergfeldt & Vahedi 2026). The brake lets go as the
# LC fires above rest (the LC inhibits parasympathetic outflow: Samuels & Szabadi 2008); VAGAL_W is sized so the brake
# and the gas share a stress heart-rate rise equally (Brindle meta-analysis, 186 studies; ENGINE-CHANGES.md change 31).
HEART_IHR, HEART_M, HEART_N = 94.0, 1.17, 0.60
# overthinking (change 31 step 3b; Immy: "it's really just overthinking"): a worry state fed by what he notices of his
# heart, explained or not, that outlasts its cause and comes back as threat (perseverative cognition: Brosschot 2006;
# Ottaviani 2016). WORRY_W is sized on Ottaviani's heart-rate effect; WORRY_IN is theory (ENGINE-CHANGES.md change 31).
WORRY_IN, WORRY_W = 1.0, 0.475   # W sized: 10 min of worry raises the heart 3.1 bpm (Ottaviani 2016 g 0.28)
# the share of a scare that reaches the brain straight, the rest now coming through the worry (Immy: "split the salt";
# the engine was sized on real people, who already overthink under stress). Sized so the average egg's TSST cortisol
# response equals its worry-off response (ENGINE-CHANGES.md change 31 step 3b addendum 4)
SCARE_DIRECT = 0.715
TAU_WORRY_UP, TAU_WORRY_DOWN = 2 / 60, 15 / 60
# the prefrontal brake on the worry (change 32; Immy: "what if it can be dynamic?"): the worry trigger is divided by how
# hard the medial prefrontal cortex is braking right now, weakened by stress (pfc_ne) and by staying up past bedtime.
# One night awake amplifies the amygdala's response 60% (Yoo 2007): 1 / (1 - 0.375) = 1.6
SLEEP_PFC = 0.375
TSST_BPM = 13.5        # the average egg's heart-rate rise at minute 15 of the engine's TSST (run 20)
VAGAL_W = 0.18          # sized: at minute 15 of the engine's TSST the brake's share = the gas's (rise +13.5 bpm)
# blood pressure (change 34b): mean arterial pressure = rest x heart rate ratio x vessel tightness ratio (stroke volume
# held: theory). 91 mmHg = 123/75 in 744 young adults (Behav Med 2021). The vessels' α1 makes 21% of resting tightness:
# IV prazosin 1 mg lowered lying pressure 18/17 mmHg with supine heart rate unchanged, and dose 1 blocks 91% of the seats
MAP_REST, VESSEL_A1 = 91.0, 0.21

CAL = None          # filled by calibrate(): scales of the fast systems, rest signals, rest knobs


def soma_5ht(ht):
    """5-HT at the raphe cell bodies from the forebrain level: a rise (transporter block) is SOMA_GAIN times as big at
    the cell bodies; a fall (less firing) is the same"""
    return 1 + SOMA_GAIN * (ht - 1) if ht > 1 else ht


def sat(x, k=1.0):
    return x / (x + k) if x > 0 else 0.0


class Chem:
    """one egg's chemistry. genes: dict overriding GENES (and rx_<receptor> amounts)."""

    def __init__(self, genes=None, hour=8.0, _calibrating=False, _no_adapt=False):
        if CAL is None and not _calibrating:
            calibrate()
        self.cal = CAL if not _calibrating else _calibrating
        self.g = dict(GENES); self.g.update(genes or {})
        self.amount = {r: self.g.get("rx_" + r, 1.0) for r in RECEPTORS}
        self.hour, self.awake, self.woke_at = (hour - 4.0) % 24, True, hour - 1.0
        self.drugs = {}                                      # name -> concentration (dose units)
        self.c = {k: 1.0 for k in ("ne", "da", "da_pfc", "ht", "ht_soma", "ach", "ha", "oxa", "oxb", "dyn", "ado", "crh",
                                   "cort", "mor", "ot", "avp", "mel", "glu_bg", "gaba", "allo", "aea", "ag2", "npy",
                                   "gal", "cyt", "bdnf", "ne_heart", "ach_heart", "ne_vessel")}
        self.c.update(mel=0.1, cyt=0.2, gal=0.0)
        self.S = 0.3                                         # process S (sleep pressure, 0..1)
        self.syn = 1.0                                       # prefrontal synapse strength (ketamine, BDNF)
        self.cea_fast = self.cea_slow = 0.0                  # amygdala excitability: reversible / rewritten (C.2 r3)
        self.cea_w = 0.0                                     # the rewrite's first stage (the message, before the protein)
        self.allo_cap = 1.0                                  # capacity to make allopregnanolone (falls with long stress)
        self.npy_reserve = 1.0                               # NPY reserve (depleted by long stress; Morgan 2000)
        self.crh_load = 0.0                                  # running average of CRH above rest (drives cea_slow)
        self.cort_total = 1.0                                # month-long cortisol total (what hair cortisol reads)
        self.calm_skill = 0.0                                # learned top-down brake on the LC (Immy's skill)
        self.values, self.seen = {}, {}                      # learned reward expectations per cue; novelty counts
        self.fear_cs = 0.0                                   # one conditioned fear (for extinction tests)
        self.inner = {}                                      # inside-the-cell tubs (INNER), filled at the first step
        self.expect = None                                   # what his body normally runs at each hour (set below)
        self.expect_hr = None                                # and his heart rate (change 31 step 3b)
        self.body_adapt = 0.0                                # what the last minutes taught him to expect on top
        self.reframe = 0.0                                   # how much he has talked himself down (reappraisal)
        self.worry_st = 0.0                                  # overthinking (change 31 step 3b)
        self.worry_adapt = 0.0                               # the heart rise he has got used to (step 3b addendum 2)
        self.shift, self.light_phase = 0.0, 0.0              # body clock behind the wall clock (h); where light is pulling it
        self.abs_t, self.sleep_start = 0.0, -99.0            # hours lived; when the last sleep began
        self.expect_ox = None                                # and what his orexin blood-pressure route normally reads
        self.inp, self.raw = {}, {k: 1.0 for k in FAST}
        self.drive, self.brake = 1.0, 0.0
        self.ref_occ, self.occ_avg = {}, {}                  # empty = receptors don't adapt (settling, reference)
        self._nm = dict(syn=0.0, int=0.0, ex=0.0)
        for _ in range(240):                                 # settle at rest for the 4 hours before `hour`
            self.step(1 / 60, awake=True)
        if not _calibrating and not _no_adapt:
            # receptors adapt against THIS egg's own normal day (sleeping 23-7), never against one moment of it
            self.ref_occ = normal_day_occupancy(genes)
            self.occ_avg = dict(self.ref_occ)
            self.expect, self.expect_ox = expected_body(genes)
            self.expect_hr = expected_heart(genes)

    # ------------------------------------------------------------ helpers
    def dev(self, r):
        """how far what receptor r does is from an average egg at rest (through its inner tub, if it has one)"""
        return self.eff(r) - self.cal["rest"].get(r, self.sig(r))

    def eff(self, r):
        """what receptor r is doing inside its cells: its inner tub if it acts through a second messenger (INNER),
        else its signal now"""
        inner = self.__dict__.get("inner")
        return inner[r] if inner and r in inner else self.sig(r)

    def _sympathetic(self):
        """how hard his body is running its alarm (the sympathetic read-out): peripheral α1 plus the orexin route above
        what it normally reads at this hour (change 26: orexin normally rises through the day; the route is not active in
        normal animals, only where orexin is overactive: Huang 2010, Li 2013), else above its resting level"""
        table = getattr(self, "expect_ox", None)
        normal = table[int(self.hour * 12) % 288] if table else self.cal["rest"].get("ox_rvlm", 1.0)
        return self.eff("a1_symp") + 3.0 * max(0.0, self.eff("ox_rvlm") - normal - 0.03)

    def _body_signal(self):
        """the body's alarm output as the body-awareness loop reads it (the read-out change 22 sized the loop on): peripheral
        α1 plus the orexin route above its 8 AM resting level. It is compared with this egg's own usual level for the
        hour, so only a surprise in the body is noticed; it is not the blood-pressure read-out (see _sympathetic)"""
        return self.eff("a1_symp") + 3.0 * max(0.0, self.eff("ox_rvlm") - self.cal["rest"].get("ox_rvlm", 1.0) - 0.03)

    def vagal_tone(self):
        """the brake on the heart (vagus): 1 at rest, let go as the LC fires above rest (change 31)"""
        return max(0.0, 1 - VAGAL_W * max(0.0, self.raw["f"] / self.cal["scale"]["f"] - 1))

    def sleep_loss(self):
        """awake past his usual bedtime pressure: 0 through a normal day, 1 by 04:00 on a night kept awake"""
        return min(1.0, max(0.0, (self.S - 0.62) / 0.1))

    def worry_brake(self):
        """how hard the prefrontal cortex is braking the worry (1 = an average egg, rested, at rest; change 32)"""
        return max(1e-3, self.pfc_ne() * (1 - SLEEP_PFC * self.sleep_loss()))

    def heart_rate(self, inp=None):
        """beats per minute: intrinsic x gas x brake (change 31). The gas is the brain's sympathetic command (the body
        signal) plus any outside push on the heart's β1 receptors (heart_push: a drug acts at the heart, not through the
        brain), relative to the command at rest"""
        rest = self.cal["rest"].get("a1_symp") or self.eff("a1_symp")          # during calibration: now is rest
        drive = (self._body_signal() + (inp or {}).get("heart_push", 0.0)) / rest
        gas = drive * self.organ_pass("b1_heart", drive)
        vagal = self.vagal_tone()
        brake = vagal * self.organ_pass("m2_heart", vagal)
        return HEART_IHR * (1 + (HEART_M - 1) * gas) * (1 - (1 - HEART_N) * brake)

    def blood_pressure(self, inp=None):
        """mean arterial pressure in mmHg (change 34b): rest x heart rate over the resting rate x vessel tightness"""
        drive = self._sympathetic() / (self.cal["rest"].get("a1_symp") or self.eff("a1_symp"))
        tight = 1 - VESSEL_A1 + VESSEL_A1 * drive * self.organ_pass("a1_vessel", drive)
        return MAP_REST * self.heart_rate(inp) / (HEART_IHR * HEART_M * HEART_N) * tight

    def organ_pass(self, r, drive):
        """the share of a drive that gets through receptor r's seats in an organ with the drugs on board (changes 33,
        34): the receptor's signal with drugs over without, at the same drive; exactly 1 with no drug"""
        if not self.drugs or drive <= 0:
            return 1.0
        lig = RECEPTORS[r][0][0]
        return self.sig(r, {lig: drive}) / self.sig(r, {lig: drive}, nodrug=True)

    def heart_felt(self, inp=None):
        """the report line (change 31 step 2): what reaches the insula of his heart, through vagal and spinal afferents
        and the NTS, within the minute: his actual heart rate"""
        return self.heart_rate(inp)

    def heart_expected(self):
        """the copy (efference copy / interoceptive prediction; Barrett & Simmons 2015): the heart rate his own commands
        (gas and brake) produce, with no outside push; a drug on the heart is not something he ordered"""
        return self.heart_rate(None)

    def heart_surprise(self, inp=None):
        """felt - copy, in bpm (interoceptive prediction error)"""
        return self.heart_felt(inp) - self.heart_expected()

    def body_surplus(self, inp=None):
        """how far his body runs above what it normally runs at this hour, plus any outside push on his heart (input
        heart_push: isoproterenol, change 30); None before his normal day is known"""
        expect = getattr(self, "expect", None)
        if expect is None:
            return None
        inp = inp or {}
        surplus = max(0.0, self._body_signal() - expect[int(self.hour * 12) % 288] + inp.get("heart_push", 0.0))
        return surplus * (1 - COPE * inp.get("breathe", 0.0))      # slow breathing slows the heart he feels (change 30)

    def heart_up(self, inp=None):
        """how far his felt heart runs above its usual rate for this hour, in TSSTs (change 31 step 3b): his own heart
        rate on a calm reference day (expected_heart); 0 before his normal day is known"""
        table = getattr(self, "expect_hr", None)
        if table is None:
            return 0.0
        x = self.hour * 12                                   # between its 5-minute bins, read smoothly
        i, f = int(x) % 288, x - int(x)
        usual = table[i] * (1 - f) + table[(i + 1) % 288] * f
        return max(0.0, self.heart_felt(inp) - usual) / TSST_BPM

    def body_noticed(self, inp=None):
        """what the bottom-up loop notices (change 31 step 3a): the heart's surprise (felt - copy, bpm), what his own
        commands did not order; slow breathing lowers what reaches him (until step 4 moves it onto the vagus). None
        before his normal day is known (no loop during calibration and the reference runs)"""
        if getattr(self, "expect", None) is None:
            return None
        inp = inp or {}
        return max(0.0, self.heart_surprise(inp)) * (1 - COPE * inp.get("breathe", 0.0))

    def body_felt(self, inp):
        """the body-awareness loop (change 22): how far his body runs above what it normally runs at this hour, less
        what the last minutes have taught him to expect (change 29), x attention to the body (his gene, a reminder from
        outside, and how vigilant his own tonic LC is, less what distraction takes away), comes back as a felt threat,
        worth less once he has talked himself down (change 30), capped smoothly"""
        surplus = self.body_noticed(inp)
        if not self.awake or BODY_LOOP == 0.0 or surplus is None:
            return 0.0
        body = max(0.0, surplus - self.__dict__.get("body_adapt", 0.0))
        vigilance = self.raw["f"] / self.cal["scale"]["f"]
        attention = self.g["body_attention"] * (1.0 + inp.get("body_reminded", 0.0)) * (1 - COPE * inp.get("distract", 0.0))
        x = BODY_LOOP * attention * vigilance * body * (1 - COPE * self.__dict__.get("reframe", 0.0))
        return BODY_CEIL * math.tanh(x / BODY_CEIL)

    def amy_out(self):
        """how much the amygdala's excitability reaches its outputs: tonic inhibition through extrasynaptic (δ)
        GABA-A receptors damps it, and loses its grip when allopregnanolone runs low after long stress
        (Antonoudiou 2022; Maguire & Mody 2008)"""
        gate = min(1.5, max(0.0, 1 - 2.0 * self.dev("gabaa_ex")))
        return (self.cea_fast + self.cea_slow) * gate

    def give(self, name, dose):
        self.drugs[name] = self.drugs.get(name, 0.0) + dose

    def stop(self, name):
        self.drugs.pop(name, None)

    def _block(self, which):
        """transporter block fraction from all drugs (0..1)"""
        b = 0.0
        for d, c in self.drugs.items():
            if which in DRUGS[d]:
                kd, mx = DRUGS[d][which]
                b = 1 - (1 - b) * (1 - mx * sat(c, kd))
        return b

    def sig(self, r, conc=None, nodrug=False):
        """signal at receptor r = amount x sum(occupancy x efficacy), natural ligands competing with drugs.
        conc: optional override of ligand concentrations (burst / dip events). nodrug: as if no drug were on board."""
        cc = self.c if conc is None else dict(self.c, **conc)
        terms, alpha = [], 1.0
        site_blocked = False
        for d, dose in ({} if nodrug else self.drugs).items():
            spec = DRUGS[d]
            site_blocked |= bool(spec.get("benzo_site_block"))
            for (rr, kind, kd, eff) in spec.get("targets", ()):
                if rr != r:
                    continue
                if kind == "pam":
                    alpha *= 1 + (eff - 1) * sat(dose, kd)
                else:
                    terms.append((dose / kd, eff))
        if r == "gabaa_syn":
            if site_blocked:
                alpha = 1.0                                  # flumazenil empties the benzodiazepine site
            elif self.g["endozepine"]:
                alpha *= 1 + self.g["endozepine"]            # (theory) a compensating endozepine
        for key, kd in RECEPTORS[r]:
            terms.append((cc[key] * alpha / kd, 1.0))
        x = sum(t for t, _ in terms)
        return self.amount[r] * sum(t * e for t, e in terms) / (1 + x)

    def nmda_state(self):
        """block of NMDA channels by all blockers: brief synaptic, repeated interneuron, sustained background"""
        p_ex = sat(self.c["glu_bg"])                          # how often background channels are open
        mg_ex = sat(self.c["glu_bg"])                         # more background = more depolarized = less Mg2+ block
        out = dict(syn=0.0, int=0.0, ex=0.0)
        for d, c in self.drugs.items():
            if "nmda" in DRUGS[d]:
                p = DRUGS[d]["nmda"]
                for mode, p_open, mg in (("syn", 0.02, p["mg_rest"]), ("int", 0.3, p["mg_rest"]), ("ex", p_ex, mg_ex)):
                    b = nmda_block(c, p["k_on"], p["k_off"], p["trap"], p_open, mg)
                    out[mode] = 1 - (1 - out[mode]) * (1 - b)
        return out

    # ------------------------------------------------------------ the fast systems at balance
    def _fast(self):
        g, i, dev, cal = self.g, self.inp, self.dev, self.cal["scale"]
        awake = 1.0 if self.awake else 0.15                  # the LC and VTA nearly stop in sleep
        threat = i.get("threat_eff", 0.0)
        net, dat, sert = (g["net"] * (1 - self._block("net")), g["dat"] * (1 - self._block("dat")),
                          g["sert"] * (1 - self._block("sert")))
        # only spontaneously active DA cells can burst; long stress silences about half of them through the
        # amygdala (Chang & Grace 2014; Moreines 2017), and they come back as the amygdala calms
        self.vta_pop = 1 / (1 + 2.0 * self.cea_slow * min(1.5, max(0.0, 1 - 2.0 * dev("gabaa_ex"))))
        self.c.setdefault("ht_soma", self.c["ht"])           # a body saved before change 24
        # raphe 5-HT with its 5-HT1A brake, which can silence the cell (why SSRIs take weeks). Solved on its own and
        # exactly (change 24): with a blocked transporter the brake is strong enough that the damped sweep below flips
        # between on and off. Firing allowed by the brake falls as firing rises, so there is one balance, found by bisection.
        fh_cap = g["ht_tone"] * (1.0 if self.awake else 0.4)

        def allowed(fh):
            return fh_cap * max(0.0, 1 - 2.0 * self.sig("ht1a_auto", {"ht_soma": soma_5ht(fh / cal["fh"] / sert)}))

        lo, hi = 0.0, fh_cap
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if allowed(mid) > mid else (lo, mid)
        fh = 0.5 * (lo + hi)
        ht_now = fh / cal["fh"] / sert
        for _ in range(40):                                  # solve the brakes at balance
            # LC: excited by threat, CRH (CRHR1), orexin (OX1R), surprise; braked by its own α2A (C.2 r1), galanin
            # (high firing only), NPY, and the learned top-down brake
            drive = (g["lc_drive"] * awake * (1 - self.calm_skill)
                     * max(0.0, 1 + 1.5 * dev("crhr1_lc") + 0.8 * dev("ox1r") + 0.8 * (threat + i.get("judged_eff", 0.0))
                           + 0.3 * i.get("surprise", 0.0)))
            brake = min(0.95, 3.0 * self.sig("a2a_auto") + 0.5 * self.sig("galr1") + 0.2 * self.sig("y1"))
            f = drive * (1 - brake)                          # firing: the brake hyperpolarizes the cell
            ne_out = f * (1 - brake)                         # release per spike: the same brake on its terminals
            # VTA DA: excited by orexin; damped by KOR (dynorphin), CRH in the VTA, cytokines, its own D2 brake
            fd = (g["da_tone"] * awake * self.vta_pop * max(0.0, 1 + 0.5 * dev("ox1r")) * max(0.0, 1 - 0.5 * dev("kor"))
                  * max(0.0, 1 - 0.4 * dev("crhr1_vta")) * max(0.0, 1 - 0.6 * dev("cytr"))
                  * (1 - 0.6 * self.sig("d2_auto")))
            ha = g["ha_tone"] * awake * max(0.0, 1 + 0.8 * dev("ox2r")) * (1 - 0.5 * self.sig("h3_auto"))
            ach = g["ach_tone"] * awake * max(0.0, 1 + 0.3 * dev("ox2r")) * max(0.0, 1 - 0.5 * dev("a1r"))
            self.raw.update(f=f, ne_out=ne_out, fd=fd, fh=fh, ha=ha, ach=ach)
            new = dict(ne=ne_out / cal["ne_out"] / net, da=fd / cal["fd"] / dat,
                       da_pfc=fd / cal["fd"] / (0.3 * dat + 0.7 * net),     # prefrontal DA is cleared mostly by NET
                       ht=ht_now, ha=ha / cal["ha"], ach=ach / cal["ach"])
            for k, v in new.items():
                self.c[k] = 0.5 * self.c[k] + 0.5 * v
        self.c["ht_soma"] = soma_5ht(self.c["ht"])
        self.drive, self.brake = drive, brake
        # background glutamate: firing + stress, cleared by EAAT, damped by CB1 (retrograde brake on release)
        glu = (g["glu_tone"] * max(0.0, 1 + 0.2 * (self.c["ne"] - 1) + 0.3 * max(0.0, self.c["crh"] - 1))
               * max(0.0, 1 - 0.3 * dev("cb1")) / g["eaat"])
        self.raw["glu_bg"] = glu
        self.c["glu_bg"] = glu / cal["glu_bg"]
        # synaptic GABA from interneurons; NMDA blockers that hold interneuron channels shut cut it (ketamine)
        self.c["gaba"] = g["gaba_tone"] * (1 - 0.7 * self.nmda_state()["int"])

    # ------------------------------------------------------------ one step
    def step(self, dt=1 / 60, **inp):
        """advance dt hours. inputs (appraisals, all optional): awake, threat (0..1), controllable, surprise,
        company (0/1), touch, rejection, judged, novelty, exercise, machine_sick (0..1), endotoxin (pulse size),
        light, safe_cue (the conditioned cue shown with nothing bad: extinction), body_reminded (0..1: his body is
        pointed out to him), heart_push (an outside push on his heart the loop can feel)"""
        g, dev = self.g, self.dev
        was = self.awake
        self.awake = inp.get("awake", True)
        if "shift" not in self.__dict__:                     # a body saved before change 27
            self.shift, self.light_phase, self.abs_t, self.sleep_start = 0.0, 0.0, 0.0, -99.0
        self.abs_t += dt
        if was and not self.awake:
            self.sleep_start = self.abs_t
        if self.awake and not was:
            self.woke_at = self.hour
            if self.abs_t - self.sleep_start >= 4.0:         # waking after a night's sleep: light reaches him now
                self.light_phase = round((((self.hour - 7.0) + 12.0) % 24.0 - 12.0) * 2) / 2   # to the half hour
        self.hour = (self.hour + dt) % 24
        err = (self.light_phase - self.shift + 12.0) % 24.0 - 12.0
        if err:
            self.shift += max(-CLOCK_ADVANCE, min(CLOCK_DELAY, err)) / 24.0 * dt
        body_hour = (self.hour - self.shift) % 24            # what time it is for his body
        # talking himself down (change 30): the new meaning forms while he reappraises and fades slowly after
        re = inp.get("reappraise", 0.0)
        tau = TAU_REFRAME_UP if re > self.__dict__.get("reframe", 0.0) else TAU_REFRAME_DOWN
        self.reframe = self.__dict__.get("reframe", 0.0)
        self.reframe += (re - self.reframe) * (1 - math.exp(-dt / tau))
        inp["threat_eff"] = (inp.get("threat", 0.0) * g["stress_gain"] * SCARE_DIRECT
                             * (0.7 if inp.get("controllable") else 1.0)
                             * (1 - COPE * inp.get("distract", 0.0)) * (1 - COPE * self.reframe))
        # overthinking (change 31 step 3b): noticing his heart up, explained or not, feeds a worry that outlasts it
        away = 1 - COPE * inp.get("distract", 0.0)
        up = self.heart_up(inp)                              # less what the last minutes made normal (addendum 2)
        fresh = max(0.0, up - self.__dict__.get("worry_adapt", 0.0))
        self.worry_adapt = self.__dict__.get("worry_adapt", 0.0)
        self.worry_adapt += (up - self.worry_adapt) * (1 - math.exp(-dt / BODY_HAB))
        drive = (WORRY_IN * fresh * g["worry_trait"] * g["body_attention"]
                 * (1.0 + inp.get("body_reminded", 0.0)) * away / self.worry_brake()) if self.awake else 0.0
        target = math.tanh(drive + inp.get("worry", 0.0)) if self.awake else 0.0
        w = self.__dict__.get("worry_st", 0.0)
        self.worry_st = w + (target - w) * (1 - math.exp(-dt / (TAU_WORRY_UP if target > w else TAU_WORRY_DOWN)))
        inp["worry_eff"] = WORRY_W * self.worry_st * (1 - COPE * self.reframe) * away
        inp["threat_eff"] += inp["worry_eff"]
        inp["body_eff"] = self.body_felt(inp)                # what his body is doing, noticed, comes back as threat
        inp["threat_eff"] += inp["body_eff"]
        surplus = self.body_noticed(inp)                     # and the expectation catches up with it (change 29)
        if surplus is not None:
            self.body_adapt = self.__dict__.get("body_adapt", 0.0)
            self.body_adapt += (surplus - self.body_adapt) * (1 - math.exp(-dt / BODY_HAB))
        # being judged is a social threat: it reaches the LC through the amygdala as well as through CRH
        inp["judged_eff"] = judged = 0.8 * inp.get("judged", 0.0) * g["stress_gain"] * SCARE_DIRECT
        self.inp = inp
        company = inp.get("company", 0.0) * g["social_gain"]

        def toward(key, target, tau):
            self.c[key] += (target - self.c[key]) * (1 - math.exp(-dt / tau))

        # drug levels (pharmacokinetics)
        for d in list(self.drugs):
            th = DRUGS[d]["t_half"]
            if th:
                self.drugs[d] *= 0.5 ** (dt / th)
                if self.drugs[d] < 1e-3:
                    del self.drugs[d]

        # sleep pressure (process S) and the clock
        if self.awake:
            self.S += (1 - self.S) * (1 - math.exp(-dt / g["sleep_rise"]))
        else:
            self.S += (0 - self.S) * (1 - math.exp(-dt / g["sleep_fall"]))
        self.c["ado"] = (0.3 + 1.4 * self.S) / g["ada"] / 0.72   # slower breakdown (ADA 22G>A) = more adenosine
        day = 0.5 + 0.5 * math.cos((body_hour - 17.0) / 24 * 2 * math.pi)     # body clock high ~5 PM
        night = not (7.0 <= body_hour < 21.5)
        toward("mel", 3.0 if night and not inp.get("light") else 0.1, TAU["mel"])

        # CRH: threat and being judged, x his gain, x amygdala excitability; NPY, CB1 and allopregnanolone (GABA-A
        # on the PVN) brake it; GR shuts it off
        cea = self.amy_out()
        # NE excites CRH cells through α1 (brainstem NE -> PVN, LC -> amygdala; C.2 rule 3: the loop is fed by his
        # own NE), so an NE surge raises CRH even with nothing threatening (yohimbine raises cortisol)
        crh_drive = (inp["threat_eff"] + judged + NE_CRH * max(0.0, dev("a1_amy"))) * (1 + cea)
        crh_t = (1 + 3.0 * crh_drive * max(0.0, 1 - 0.5 * dev("y1")) * max(0.0, 1 - 0.5 * dev("cb1"))
                 * max(0.0, 1 - 0.8 * dev("gabaa_ex")) / (1 + 4.0 * g["gr_feedback"] * max(0.0, dev("gr"))))
        toward("crh", max(0.2, crh_t), TAU["crh"])

        # cortisol: low around sleep onset, rising through the night to a peak at waking, + the waking rise + CRH
        h = body_hour
        circ = 0.3 + 1.3 * ((h / 7.5) ** 2 if h < 7.5 else ((24 - h) / 16.5) ** 1.5)
        since = (self.hour - self.woke_at) % 24
        car = 0.6 * math.exp(-((since - 0.6) / 0.35) ** 2) if self.awake else 0.0
        # CRH -> ACTH -> cortisol, sized so 15 minutes of being judged adds ~40% of the waking peak (TSST
        # responders roughly double their cortisol; Kirschbaum 1993, Dickerson & Kemeny 2004)
        toward("cort", circ + car + 3.5 * max(0.0, self.c["crh"] - 1), TAU["cort"])
        self.cort_total += (self.c["cort"] - self.cort_total) * (1 - math.exp(-dt / (24 * 30)))

        # orexin cells: day drive while awake, excited by NE (α1 on the cells), CRH, exploring; adenosine (A1)
        # inhibits them; one release, two peptides (A stable, B fast)
        if self.awake:
            sleep_loss = self.sleep_loss()
            x = (g["ox_cells"] * (0.5 + 0.5 * day)
                 * max(0.0, 1 + 3.0 * g["ne_ox_link"] * dev("a1_ox") + 0.3 * max(0.0, self.c["crh"] - 1)
                       + 0.3 * inp.get("novelty", 0.0) - 1.5 * sleep_loss * self.sig("a2_ox"))
                 * max(0.0, 1 - 0.5 * dev("a1r")))
            x += g["ox_cells"] * sum(DRUGS[d].get("ox_drive", 0) * sat(c) for d, c in self.drugs.items())
        else:
            x = 0.05 * g["ox_cells"]
        self.raw["x_ox"] = x
        x /= self.cal["scale"]["x_ox"]
        toward("oxa", x, TAU["oxa"]); toward("oxb", x, TAU["oxb"])
        # dynorphin: packed with orexin, plus stress
        toward("dyn", 0.6 * x + 0.4 + 0.8 * max(0.0, self.c["crh"] - 1), TAU["dyn"])

        # social and comfort chemicals
        rej = inp.get("rejection", 0.0)
        toward("mor", g["mor_tone"] * max(0.1, 1 + 0.8 * company + 0.5 * inp.get("touch", 0.0) - 0.4 * rej
                                          - 0.3 * inp.get("alone_long", 0.0)), TAU["mor"])
        toward("ot", g["ot_tone"] * (1 + company + inp.get("touch", 0.0)), TAU["ot"])
        toward("avp", g["avp_tone"] * (1 + 0.3 * company + 0.5 * max(0.0, self.c["crh"] - 1)), TAU["avp"])

        # NPY: released with NE under stress from a reserve that empties with long stress, refills over hours. What
        # each release carries scales with how much NPY his cells make (low-NPY people release less under stress;
        # Morgan 2000), so the gene sets the reserve and the burst alike
        frac = 0.6 * max(0.0, self.c["ne"] - 1.3) * self.npy_reserve
        self.npy_reserve = min(1.0, max(0.0, self.npy_reserve + dt * (1 - self.npy_reserve) / TAU["npy_refill"]
                                        - dt * 0.4 * frac))
        toward("npy", g["npy_tone"] * (0.8 + 0.2 * self.npy_reserve + frac), TAU["npy"])
        # galanin: only released when the LC fires hard (dense-core vesicles need high-frequency firing, ~3x tonic)
        toward("gal", 1.5 * max(0.0, self.raw["f"] / self.cal["scale"]["f"] - 3.0), TAU["gal"])

        # allopregnanolone: up with acute stress, capacity falls with long stress
        self.allo_cap += ((1 - 0.6 * min(1.0, self.crh_load)) - self.allo_cap) * (1 - math.exp(-dt / TAU["allo_cap"]))
        toward("allo", g["allo_tone"] * self.allo_cap * (1 + 0.5 * max(0.0, self.c["crh"] - 1)), TAU["allo"])
        # endocannabinoids: anandamide is a gate that falls with stress; 2-AG is made on demand to end it
        toward("aea", g["ecb_tone"] / g["faah"] * max(0.2, 1 - 0.25 * max(0.0, self.c["crh"] - 1)), TAU["aea"])
        toward("ag2", g["ecb_tone"] * (1 + 0.4 * max(0.0, self.c["crh"] - 1)), TAU["ag2"])

        # cytokines: his machine's health (Immy 2026-10-22) and an endotoxin pulse
        if inp.get("endotoxin"):
            self.c["cyt"] += inp["endotoxin"]
        toward("cyt", g["sick_gain"] * (0.2 + 2.0 * inp.get("machine_sick", 0.0)), TAU["cyt"])

        # the fast systems at balance
        self._fast()
        self._nm = self.nmda_state()
        # inside-the-cell tubs follow their receptors (change 21)
        if "inner" not in self.__dict__:
            self.inner = {}                                  # a body saved before the tubs existed
        k_in = 1 - math.exp(-dt / TAU_INNER)
        for r in INNER:
            now = self.sig(r)
            self.inner[r] = now if r not in self.inner else self.inner[r] + (now - self.inner[r]) * k_in

        # BDNF: activity (exercise, a glutamate surge when interneurons are held shut), lowered by high GR; made
        # fast, cleared slowly
        surge = max(0.0, 1 - self.c["gaba"] / g["gaba_tone"])
        tgt = (g["bdnf_tone"] * (1 + 0.8 * inp.get("exercise", 0.0) + 3.0 * surge)
               * max(0.2, 1 - 0.8 * max(0.0, dev("gr"))))
        toward("bdnf", tgt, TAU["bdnf_up"] if tgt > self.c["bdnf"] else TAU["bdnf"])
        # prefrontal synapses follow BDNF: new spines within a day, lost over days
        tgt = 1 + 2.0 * dev("trkb")
        self.syn += (tgt - self.syn) * (1 - math.exp(-dt / (TAU["syn_up"] if tgt > self.syn else TAU["syn"])))
        self.syn = max(0.5, min(1.6, self.syn))

        # amygdala excitability (C.2 rule 3): the fast part follows CRH and undoes itself in ~30 min. When CRH stays
        # up for days (its 3-day average above 0.1) the slow part is rewritten: a gene program, so two stages (the
        # message, then the protein), each on a 4-day clock. It holds for about a week after the stress ends and is
        # mostly gone by two weeks, whatever CRH does. Scaled so ~4 weeks of daily stress silences about half the
        # DA cells (vta_pop), the size Chang & Grace 2014 and Moreines 2017 measured after chronic mild stress.
        self.cea_fast += (0.8 * max(0.0, dev("crhr1_amy")) - self.cea_fast) * (1 - math.exp(-dt / TAU["cea_fast"]))
        self.crh_load += (max(0.0, self.c["crh"] - 1) - self.crh_load) * (1 - math.exp(-dt / TAU["crh_load"]))
        k_slow = 1 - math.exp(-dt / TAU["cea_slow"])
        self.cea_w += dt * CEA_RATE * max(0.0, self.crh_load - 0.1) - self.cea_w * k_slow
        self.cea_slow += (self.cea_w - self.cea_slow) * k_slow

        # receptors that adapt to long use (BDNF sets the speed): a day-long average of how full each one is,
        # against how full it is on this egg's normal day
        speed = 0.5 + self.sig("trkb")
        for r, tau in (ADAPT.items() if self.ref_occ else ()):
            if r in FIRING_SETPOINT:
                fire = self.raw[FIRING_SETPOINT[r]]
                self.occ_avg[r] += (fire - self.occ_avg[r]) * (1 - math.exp(-dt / 24.0))
                # fewer autoreceptors while the cell fires below its normal, more while above
                self.amount[r] *= math.exp(dt * speed / tau * (self.occ_avg[r] / self.ref_occ[r] - 1))
                self.amount[r] = min(2.0, max(0.2, self.amount[r]))
                continue
            occ = self.sig(r) / max(1e-6, self.amount[r])
            self.occ_avg[r] += (occ - self.occ_avg[r]) * (1 - math.exp(-dt / 24.0))
            # partial homeostasis: amounts move half-way (in log terms) to undo a lasting change in how full they are
            target = self.g.get("rx_" + r, 1.0) * min(1.5, max(0.3, (self.ref_occ[r] / max(1e-6, self.occ_avg[r])) ** 0.5))
            self.amount[r] += (target - self.amount[r]) * (1 - math.exp(-dt * speed / tau))

        # one conditioned fear: shown safely it fades, faster with CB1 and NPY (extinction)
        if inp.get("safe_cue"):
            self.fear_cs -= dt * 0.6 * self.fear_cs * max(0.05, 1 + dev("cb1") + 0.5 * dev("y1"))
        # NE from the cardiac nerves (bookkeeping: heart_rate() passes the drive with any push itself)
        self.c["ne_heart"] = self._body_signal() / (self.cal["rest"].get("a1_symp") or self.eff("a1_symp"))
        self.c["ach_heart"] = self.vagal_tone()
        self.c["ne_vessel"] = self._sympathetic() / (self.cal["rest"].get("a1_symp") or self.eff("a1_symp"))
        return self.knobs()

    # ------------------------------------------------------------ events (phasic parts)
    def reward_event(self, cue, value=1.0, alpha=0.3):
        """a reward (or its omission, value 0) after a cue: DA burst / dip = prediction error (Schultz; TD learning)
        plus a novelty bonus that fades with repetition (Kakade & Dayan 2002)"""
        v = self.values.get(cue, 0.0)
        n = self.seen.get(cue, 0)
        delta = value * self.g["reward_gain"] + 0.5 / (1 + n) - v
        self.values[cue] = v + alpha * (value * self.g["reward_gain"] - v)
        self.seen[cue] = n + 1
        rel = self.burst_factor()
        peak = self.c["da"] * (1 + 8.0 * delta * rel) if delta > 0 else self.c["da"] * max(0.02, 1 + delta)
        return dict(delta=delta, da_peak=peak, d1=self.sig("d1", {"da": peak}), d2=self.sig("d2", {"da": peak}),
                    d1_rest=self.sig("d1"), d2_rest=self.sig("d2"))

    def burst_factor(self):
        """how big a reward burst is now (1 = an average egg at rest): release per burst is cut by KOR, CRH in the VTA
        and cytokines (the same factors as tonic firing), and only the active DA cells can burst"""
        return (self.vta_pop * max(0.0, 1 - 0.5 * self.dev("kor")) * max(0.0, 1 - 0.4 * self.dev("crhr1_vta"))
                * max(0.0, 1 - 0.6 * self.dev("cytr")))

    def burst_release(self):
        """how much NE an LC burst to something unexpected releases (1 = an average egg at rest). A burst is driven
        and too fast for its own brake (C.2 rule 1), but two things shrink it: a brake already standing from loud
        background firing, and a cell too quiet to burst at all (drowsy). So bursts are biggest at a middle
        background level (Aston-Jones, Rajkowski & Cohen 1999)."""
        exc = sat(self.drive ** 3, 0.3 ** 3) / self.cal["exc"]
        return exc * (1 - self.brake) / (1 - self.cal["brake"])

    def _write(self, b1):
        """how hard a moment is written (K7): M1, β1, GR; open-channel NMDA block and strong GABA-A cut it"""
        s = self.eff
        return (s("m1") * (1 + b1 + 0.5 * s("gr")) * (1 - 0.8 * self._nm["syn"])
                * max(0.0, 1 - 1.5 * max(0.0, self.dev("gabaa_syn"))))

    def _lr(self, r):
        """log ratio of receptor r's signal to its rest signal, softened so an empty receptor stays finite"""
        rest = self.cal["rest"].get(r, 1.0)
        return math.log((self.eff(r) + 0.25 * rest) / (1.25 * rest))

    def ne_burst(self):
        """an LC burst: its release, the pupil change, β1 at the peak and how hard that moment is written"""
        rel = self.burst_release()
        b1 = self.sig("b1", {"ne": self.c["ne"] * (1 + 3 * rel)})
        return dict(release=rel, pupil_evoked=rel, b1=b1, write=self._write(b1))

    # ------------------------------------------------------------ knobs (E): what the receptors change in him
    def pfc_ne(self):
        """NE's part of the prefrontal network (1 = an average egg at rest), from Immy's chain. α2A cuts cAMP through
        Gi with spare receptors (2/3 of its most at rest: moderate NE engages α2A), β1 adds to it; HCN channels open
        as cAMP rises (two cAMP sites working together: Hill 2) and the network weakens; α1 -> PKC needs DAG and Ca2+
        together (Hill 2). The inverted U is written nowhere: it comes from α2A's help leveling off while α1 and β1
        keep climbing with NE"""
        rest = self.cal["rest"]
        def rel(r):
            return self.eff(r) / max(1e-9, rest.get(r, self.sig(r)))
        sa, sa0 = self.eff("a2a_pfc"), rest.get("a2a_pfc", self.sig("a2a_pfc"))
        gi, gi0 = sa / (sa + 0.5 * sa0), 2 / 3
        camp = ((1 + PFC_BETA * rel("b1_pfc")) * (1 - PFC_GI_MAX * gi)) / ((1 + PFC_BETA) * (1 - PFC_GI_MAX * gi0))
        hcn = 2 / (1 + camp ** 2)
        pkc = (1 + PFC_PKC) / (1 + PFC_PKC * rel("a1_pfc") ** 2)
        return hcn * pkc

    def knobs(self):
        s = self.eff
        nm = self._nm
        u_ne = 0.5 * self.pfc_ne()                                    # Immy's chain (α2A / β1 cAMP, α1 PKC)
        u_da = 4 * s("d1_pfc") * (1 - s("d1_pfc"))
        noise = sat(self.c["glu_bg"]) * (1 - nm["ex"])
        f_rel = self.raw["f"] / self.cal["scale"]["f"]
        burst = self.burst_release()
        k = {}
        k["deliberate"] = u_ne * u_da * self.syn * (1 - 0.5 * noise) * (1 - 0.6 * nm["syn"])
        # the LC's two modes (Aston-Jones & Cohen 2005): useful bursts help; background above the alert level adds
        # false alarms
        k["focus"] = burst / max(1.0, f_rel)
        k["trust_input"] = s("m1")
        k["explore"] = f_rel
        steady = (0.5 * s("d2") + s("d3") + 0.5 * s("ox1r") - 0.6 * s("kor")) * self.syn
        k["wanting"] = steady * ((1 - WANT_BURST) + WANT_BURST * self.burst_factor())
        k["patience"] = s("ht1a_post")
        k["write"] = self._write(s("b1"))
        # wake drive: each system counts by how far its receptor signal moved from rest, as a log ratio, so a nearly
        # full receptor (D2, A1) and a nearly empty one (D1) both report their changes (theory: weights by how
        # strongly each system's drugs and lesions move wakefulness)
        k["awake_z"] = sum(w * self._lr(r) for r, w in WAKE.items())
        k["awake"] = 1 / (1 + math.exp(-(k["awake_z"] - self.cal["awake_bias"])))
        k["comfort"] = s("mor")
        k["dysphoria"] = s("kor")
        # anxious tone: what drives the amygdala times the gain its inhibition leaves (shunting inhibition sets gain,
        # Mitchell & Silver 2003). Each brake counts by how far it moved from rest (log ratio, as for wake), so low
        # GABA or NPY makes every push bigger instead of adding a fixed amount, and a benzodiazepine is as strong
        # as it is in people
        amy_drive = (1.2 * s("crhr1_amy") + 1.0 * s("a1_amy") + 1.5 * self.amy_out() + 0.4 * s("ox1r")
                     + 0.3 * s("ht2c") + 0.2 * s("v1b"))
        k["anxious"] = amy_drive * math.exp(-sum(w * self._lr(r) for r, w in AMY_BRAKES.items()))
        k["recognize"] = s("oxtr") * math.sqrt(max(0.0, s("v1a")))
        k["warmth"] = s("oxtr")
        k["liking"] = s("mor") + 0.5 * s("cb1")
        k["noise"] = noise
        k["sick"] = s("cytr")
        k["sympathetic"] = self._sympathetic() * (1 - COPE * self.inp.get("breathe", 0.0))   # slow breathing (change 30)
        k["heart_rate"] = self.heart_rate(self.inp)                                        # bpm (change 31)
        k["heart_surprise"] = self.heart_surprise(self.inp)                                # felt - copy (step 2)
        k["map"] = self.blood_pressure(self.inp)                                           # mmHg (change 34b)
        k["pupil"] = f_rel
        k["burst"] = burst
        k["startle"] = (1 + self.amy_out()) * (0.5 + s("crhr1_amy") + s("a1_amy"))
        k["plasticity"] = s("trkb")
        k["social_drive"] = (s("d3") + s("oxtr")) * max(0.0, 1 - 0.8 * self.dev("cytr"))
        k["fear_cs"] = self.fear_cs * (1 + self.amy_out())
        return k


_REF = {}


def normal_day_occupancy(genes=None):
    """how full each adapting receptor is on average over a normal day (awake 7-23) for an egg with these genes;
    cached per gene set"""
    key = tuple(sorted((genes or {}).items()))
    if key not in _REF:
        e = Chem(genes, hour=7.0, _no_adapt=True)
        acc, n = {r: 0.0 for r in ADAPT}, 24 * 12
        for i in range(2 * n):
            e.step(1 / 12, awake=7.0 <= e.hour < 23.0)
            if i >= n:
                for r in ADAPT:
                    v = e.raw[FIRING_SETPOINT[r]] if r in FIRING_SETPOINT else e.sig(r) / max(1e-6, e.amount[r])
                    acc[r] += v / n
        _REF[key] = acc
    return dict(_REF[key])


_EXPECT = {}


def population_ox():
    """what the orexin blood-pressure route normally reads at each hour of a calm day in an average egg (288 five-minute
    bins): the population's normal, which "overactive orexin" is measured against (change 26). Cached."""
    if "ox" not in _EXPECT:
        n = 24 * 12
        e = Chem(None, hour=7.0, _no_adapt=True)
        ox = [0.0] * 288
        for i in range(2 * n):
            e.step(1 / 12, awake=7.0 <= e.hour < 23.0)
            if i >= n:
                ox[int(e.hour * 12) % 288] = e.eff("ox_rvlm")
        _EXPECT["ox"] = ox
    return list(_EXPECT["ox"])


def expected_body(genes=None):
    """what his body normally runs at each hour of a calm day, for an egg with these genes: the body signal the loop
    reads (288 five-minute bins, the second of two normal days, awake 7-23, the body-awareness loop out of it). The loop
    compares his body against its own usual level, so only a surprise in his body is noticed. Returns (that table, the
    population's orexin-route table). Cached per gene set."""
    key = tuple(sorted((genes or {}).items()))
    if key not in _EXPECT:
        n = 24 * 12
        e = Chem(genes, hour=7.0, _no_adapt=True)
        sym, hr = [0.0] * 288, [0.0] * 288
        for i in range(2 * n):
            e.step(1 / 12, awake=7.0 <= e.hour < 23.0)
            if i >= n:
                sym[int(e.hour * 12) % 288] = e._body_signal()
                hr[int(e.hour * 12) % 288] = e.heart_rate()
        _EXPECT[key] = sym
        _EXPECT[("heart",) + key] = hr
    return list(_EXPECT[key]), population_ox()


def expected_heart(genes=None):
    """his heart rate at each hour of the same calm reference day (change 31 step 3b): what he is used to feeling"""
    expected_body(genes)
    return list(_EXPECT[("heart",) + tuple(sorted((genes or {}).items()))])


def calibrate(rounds=200, tol=1e-6):
    """settle an average egg at rest (8 AM, awake) until its fast systems sit at 1; record rest receptor signals,
    the standing LC brake, how ready the LC is to burst, and the bias that puts 'awake' at 0.85 at rest.
    Loops through rest (NE -> CRH -> LC) make a plain repeat swing back and forth, so each round moves the scales
    only half-way (in log terms) and calibration fails loudly unless every fast system ends within tol of 1."""
    global CAL
    cal = dict(scale={k: 1.0 for k in FAST}, rest={}, brake=0.0, exc=1.0, awake_bias=0.0)
    for _ in range(rounds):
        egg = Chem(hour=8.0, _calibrating=cal)
        worst = max(abs(egg.raw[k] / cal["scale"][k] - 1) for k in FAST)
        if cal["rest"] and worst < tol:
            break
        for k in FAST:
            cal["scale"][k] = math.sqrt(cal["scale"][k] * max(1e-6, egg.raw[k]))
        cal["rest"] = {r: egg.sig(r) for r in RECEPTORS}
        cal["brake"], cal["exc"] = egg.brake, sat(egg.drive ** 3, 0.3 ** 3)
    else:
        raise RuntimeError(f"calibration did not settle: fast systems still {worst:.2g} from rest")
    egg = Chem(hour=8.0, _calibrating=cal)
    z = egg.knobs()["awake_z"]
    cal["awake_bias"] = z - math.log(0.85 / 0.15)
    CAL = cal
    ref = Chem(hour=8.0)
    CAL["knobs"] = ref.knobs()
    return CAL


# ---------------------------------------------------------------- second-scale models
def nmda_block(dose, k_on, k_off, trap, p_open, mg):
    """steady block of NMDA channels by an open-channel blocker. It can only enter an OPEN channel (uncompetitive:
    more opening = more block), Mg2+ at rest keeps some blockers out (mg = share that gets past), it leaves an open
    channel at k_off, and a closed channel traps it (trap = share that can't leave while closed).
    Lipton 2006: memantine = low affinity, fast off, so it blocks channels held open and spares brief synaptic ones."""
    on = p_open * k_on * dose * mg
    off = p_open * k_off + (1 - p_open) * k_off * (1 - trap)
    return on / (on + off) if on > 0 else 0.0


def lc_spikes(rate_hz, burst_at=None, seconds=8.0, dt=0.001, latency=0.1, rise=0.8, gain=1.0, standing=0.0):
    """millisecond model of one LC cell: each spike releases NE; released NE turns on the α2A brake only after a
    latency (58-154 ms) and it peaks ~0.9 s after the spike (GIRK). The brake lowers both release and firing. A burst
    is 3 spikes 40 ms apart. Returns spike times, per-spike release (1 = an unbraked spike) and the brake trace."""
    n = int(seconds / dt)
    kt = np.arange(0, 4.0, dt)
    tt = np.clip(kt - latency, 0, None)
    kern = np.where(kt > latency, (tt / rise) * np.exp(1 - tt / rise), 0.0)    # alpha function, peak at L + rise
    release = np.zeros(n)
    spikes, rel, brake = [], [], np.zeros(n)
    phase = 0.0
    burst = set()
    if burst_at is not None:
        b0 = int(burst_at / dt)
        burst = {b0, b0 + int(0.04 / dt), b0 + int(0.08 / dt)}
    for t in range(n):
        lo = max(0, t - len(kern) + 1)
        b = min(0.95, standing + gain * 0.25 * float(np.dot(release[lo:t + 1][::-1], kern[:t + 1 - lo])))
        brake[t] = b
        phase += rate_hz * dt * (1 - b)                       # the brake hyperpolarizes the cell: fewer spikes
        fire = t in burst
        if not fire and phase >= 1.0:
            fire = True
        if fire:
            phase = 0.0                                       # every spike resets the pacemaker (its after-hyperpolarization)
        if fire:
            release[t] = 1 - b
            spikes.append(t * dt); rel.append(1 - b)
    return np.array(spikes), np.array(rel), brake
