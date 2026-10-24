"""The 34-series receptor additions and their tissue connections.

This is an exploratory, normalized physiology model. Binding priors are kept with
their provenance in receptor_parameters.json. Coupling strengths and unsupported
kinetics are explicit hypotheses, frozen before the assembled-system evaluation.
Brain, peripheral nerve, gut and endocrine pools have distinct names. Drug exposure
is an explicit relative-compartment approximation, not a pharmacokinetic model.
"""
import json
import math
from pathlib import Path

PARAMETERS = json.loads(Path(__file__).with_name('receptor_parameters.json').read_text())
SITES = PARAMETERS['receptors']
RECEPTORS = {name: [tuple(x) for x in row['ligands']] for name, row in SITES.items()}
PERIPHERAL_OLD = {'b1_heart', 'm2_heart', 'a1_vessel'}
OLD_B1_EPI_KD = 10 * 10**(-5.15 + 5.74)  # Baker 2010; absolute NE anchor remains theory.
EPI_REST = .05
COUPLING_SCALE = 1.0                 # fixed .5/2 sensitivity is report-only, after evaluation
CARDIAC_B2 = .20                     # functional prior, not a receptor-density estimate
VESSEL_B2 = .40
VESSEL_A2B = .10
BARO_TAU, BARO_RESET = 3 / 60, 6.0
BARO_SYMP, BARO_VAGAL = .08, .028     # per mmHg; approximate small-signal 2 bpm/mmHg
NHR_TAU, CYTOKINE_RECEPTOR_TAU = 6.0, 1.0
NIC_DESENSITIZE, NIC_RECOVER, NIC_UPREGULATE = 100 / 3600, 10 / 60, 24 * 7
ORGAN_MESSENGERS = ('b1_heart', 'b2_heart', 'a1_vessel', 'b2_vessel')

# Signed deviations from the calibrated reference. Every coefficient is a declared
# coarse-model coupling prior, not an effect size inferred from binding affinity.
EFFECTS = {
    'lc': {'nic42': .25, 'ht2a_lc': -.25, 'gabab': -.25, 'nop': -.25, 'crhr2': -.25},
    'ne_release': {'a2c': -.5},
    'da': {'nic42': .25, 'm4': -.25},
    'ach': {'m2_brain': -.5},
    'glu': {'ht2a_cortex': .25, 'nic7': .25, 'mglur2': -.25, 'mglur3': -.25, 'gabab': -.5},
    'pfc': {'d4': .1, 'd5': -.1},
    'wake': {'h2_brain': .25, 'nic42': .25, 'gabab': -.25, 'nop': -.25},
    'memory': {'b2_brain': .25},
    'plasticity': {'nmda_syn': .1, 'mglur5': .1, 'galr2': .1},
    'orexin': {'ghsr': .25, 'lepr': -.25},
    'clock': {'ht7': .1, 'mt2': .1},
    'immune': {'cb2': -.25},
    'npy_release': {'y2': -.5},
    'threat': {'cckb': 1.0, 'asic1a': 1.0},
    'pain': {'trpv1': 1.0, 'nk1': 1.0, 'dor': -.25, 'nop': -.25, 'glyr': -.25},
    'nausea': {'ht3_gut': 1.0, 'nk1': 1.0},
    'comfort': {'dor': .25},
    'hunger': {'ghsr': 1.0, 'lepr': -1.0, 'ccka': -1.0},
    'motility': {'m3_gut': .25, 'ghsr': .25, 'ccka': -.25},
    'metabolism': {'tr_alpha': .25, 'tr_beta': .25, 'b3_adipose': .25, 'lepr': .1},
    'reproductive': {'er_alpha': .1, 'er_beta': .1, 'ar': .1, 'pr': -.1, 'prlr': -.1},
    'motor': {'ampa': .25, 'glyr': -.25},
}
INITIAL = {
    'adrenaline': EPI_REST, 'ne_peripheral': 1., 'ach_ganglion': 1., 'ach_gland': 1., 'ach_gut': 1.,
    'ht_gut': 1., 'ha_gut': 1., 'aea_peripheral': 1., 'ag2_peripheral': 1.,
    'enkephalin': 1., 'nociceptin': 1., 'urocortin': 1., 'cck_brain': 1., 'cck_gut': 1.,
    'substance_p': 1., 'glycine': 1., 'glu_syn': 1., 'thyroid': 1., 'estrogen': 1.,
    'androgen': 1., 'progesterone': 1., 'ghrelin': 1., 'leptin': 1., 'prolactin': 1.,
    'da_tida': 1., 'proton': 10**(6.5 - (6.1 + math.log10(24 / (0.03 * 40)))),
}


def bracketed_root(fn, low, high, tolerance=1e-10):
    """Solve a continuous bracketed scalar residual, failing loudly if unresolved."""
    left, right = fn(low), fn(high)
    if not all(math.isfinite(v) for v in (left, right)) or left > 0 or right < 0:
        raise ArithmeticError('feedback residual has no finite root bracket')
    if abs(left) <= tolerance:
        return low
    if abs(right) <= tolerance:
        return high
    for _ in range(80):
        mid = (low + high) / 2
        value = fn(mid)
        if not math.isfinite(value):
            raise ArithmeticError('nonfinite feedback residual')
        if abs(value) <= tolerance:
            return mid
        if value < 0:
            low = mid
        else:
            high = mid
    raise ArithmeticError('feedback residual did not converge')


def install(receptors, drugs):
    """Install data before calibration; existing anchor targets keep their original values."""
    overlap = receptors.keys() & RECEPTORS.keys()
    if overlap:
        raise ValueError(f'duplicate receptor definitions: {overlap}')
    receptors.update(RECEPTORS)
    receptors['b1_heart'].append(('adrenaline', OLD_B1_EPI_KD))
    for name, profile in PARAMETERS['drug_profiles'].items():
        if profile['new']:
            if name in drugs:
                raise ValueError(f'duplicate drug: {name}')
            drugs[name] = {k: v for k, v in profile.items() if k in ('t_half', 'sert', 'exposure')}
            drugs[name]['targets'] = []
        existing = {x[0] for x in drugs[name].get('targets', [])}
        drugs[name].setdefault('targets', []).extend(tuple(t) for t in profile['targets'] if t[0] not in existing)
        drugs[name].setdefault('exposure', {'brain': 1.0, 'peripheral': 1.0})


def compartment(receptor):
    return SITES[receptor]['compartment'] if receptor in SITES else (
        'peripheral' if receptor in PERIPHERAL_OLD else 'brain')


def slow_receptors():
    result = {r: (NHR_TAU if s['coupling'] == 'nuclear' else CYTOKINE_RECEPTOR_TAU)
              for r, s in SITES.items() if s['coupling'] in ('nuclear', 'JAK_STAT')}
    result['gabab'] = 1 / 60  # coarse one-minute metabotropic inhibitory response, theory
    return result


def messenger_receptors():
    return tuple(dict.fromkeys(ORGAN_MESSENGERS + tuple(
        r for r, s in SITES.items() if s['coupling'] in ('Gs', 'Gq'))))


class ReceptorBody:
    def init_receptor_body(self):
        self.c.update(INITIAL)
        self.drug_exposure = {}       # name -> {brain, peripheral}; user-controlled model experiment
        self.nic_available = {r: 1.0 for r in ('nic42', 'nic7', 'nic_ganglion')}
        self.baro_error = self.baro_adapt = 0.0
        self.pco2, self.brain_ph = 40.0, 6.1 + math.log10(24 / 1.2)
        self.ventilation = 1.0
        self.energy_store = 1.0
        self.command_inner = {}     # matched-delay counterfactual, never actual drug-exposed history

    def receptor_exposure(self, drug, receptor, spec):
        place = compartment(receptor)
        return self.drug_exposure.get(drug, {}).get(place, spec.get('exposure', {}).get(place, 1.0))

    def receptor_effect(self, pathway):
        return COUPLING_SCALE * sum(w * self.dev(r) for r, w in EFFECTS[pathway].items())

    def receptor_factor(self, pathway):
        return math.exp(self.receptor_effect(pathway))

    def receptor_gate(self, receptor, signal, cc, *, nodrug=False):
        """Channel coincidence, not a second competitive binding site at the same pocket."""
        if receptor == 'nmda_syn':
            gly = max(0., cc['glycine'])
            glu = max(0., cc['glu_syn'])
            block = 0. if nodrug else self.nmda_state()['syn']
            return signal * gly / (1 + gly) * glu / (1 + glu) * (1 - block)
        if receptor == 'trpv1':
            # Independent thermal and acid gates. 43 C half-activation and a 2 C
            # slope are coarse priors; this is not a temperature injury model.
            heat = 1 / (1 + math.exp(-(self.inp.get('temperature_c', 37.) - 43.) / 2))
            acid = max(0., cc['proton']) / (1 + max(0., cc['proton']))
            amount = self.amount[receptor]
            bound = signal / amount if amount else 0.
            return amount * (1 - (1 - bound) * (1 - heat) * (1 - acid))
        return signal

    def relative_signal(self, receptor, *, nodrug=False, conc=None):
        now = self.sig(receptor, conc, nodrug=nodrug)
        if not nodrug:
            now *= self.nic_available.get(receptor, 1.)
        ref = self.cal['rest'].get(receptor)
        if ref is None:
            # Calibration reference is a unit-amount receptor, never this egg's
            # mutated abundance. These sites have nonzero reference agonist.
            ref = self.sig(receptor, conc, nodrug=True) / max(1e-12, self.amount[receptor])
        return now / max(1e-12, ref)

    def organ_drive(self, receptor, concentration, *, nodrug=False, reference=None):
        """Direct activation normalized to a fixed unit receptor, with no inverse saturation."""
        return self.sig(receptor, concentration, nodrug=nodrug) / self.organ_reference(receptor, reference)

    def organ_reference(self, receptor, reference=None):
        """Fixed reference independent of this subject's amount, drugs and current ligand pool."""
        reference = reference if reference is not None else {
            ligand: EPI_REST if ligand == 'adrenaline' else 1.
            for ligand, _ in self.receptor_definitions[receptor]}
        q = sum(reference.get(ligand, 0.) / kd for ligand, kd in self.receptor_definitions[receptor])
        if q <= 0 or not math.isfinite(q):
            raise ValueError('organ normalization needs positive finite reference activation')
        return q / (1 + q)

    def organ_signals(self, *, commanded=False):
        """Current receptor inputs; calling this does not advance any messenger state."""
        rest = self.cal['rest'].get('a1_symp') or self.eff('a1_symp')
        concentrations = {
            'ne_heart': self.terminal_command('heart', commanded=commanded)
                        + (0. if commanded else self.inp.get('heart_push', 0.) / rest),
            'ne_vessel': self.terminal_command('vessel', commanded=commanded),
            'adrenaline': EPI_REST if commanded else self.c['adrenaline'],
        }
        return {r: self.sig(r, concentrations, nodrug=commanded) for r in ORGAN_MESSENGERS}

    def organ_activation(self, receptor, *, commanded=False):
        history = self.command_inner if commanded else self.inner
        if receptor in history:
            return history[receptor]
        return self.organ_signals(commanded=commanded)[receptor]

    def terminal_command(self, which, *, commanded=False):
        rest = self.cal['rest'].get('a1_symp') or self.eff('a1_symp')
        drive = (self._body_signal() if which == 'heart' else self._sympathetic()) / rest
        ganglion = self.relative_signal('nic_ganglion', nodrug=commanded)
        drive *= ganglion
        if not commanded:
            drive *= math.exp(BARO_SYMP * self.baro_error)
        # Shared inherited NET expression, separate tissue concentration/clearance.
        clearance = self.g['net'] * (1 - (0. if commanded else self._block('net', compartment='peripheral')))
        drive /= clearance
        receptor, ligand = ('a2a_heart_auto', 'ne_heart') if which == 'heart' else ('a2a_vessel_auto', 'ne_vessel')
        kd = self.receptor_definitions[receptor][0][1]
        reference = drive / (drive + kd) if drive else 0.
        current = self.sig(receptor, {ligand: drive}, nodrug=commanded)
        # The reference includes ordinary autoregulation already absorbed in the
        # old nerve-command calibration. Altered abundance and drugs do not cancel.
        return drive * (1 + reference) / (1 + current)

    def body_vagal(self, *, commanded=False):
        value = self.vagal_tone() * self.relative_signal('nic_ganglion', nodrug=commanded)
        return value if commanded else value * math.exp(-BARO_VAGAL * self.baro_error)

    def cardiac_gas(self, *, commanded=False):
        b1 = self.organ_activation('b1_heart', commanded=commanded) / self.organ_reference('b1_heart')
        b2 = self.organ_activation('b2_heart', commanded=commanded) / self.organ_reference('b2_heart')
        return (1 - CARDIAC_B2) * b1 + CARDIAC_B2 * b2

    def vascular_tightness(self, alpha1_weight, *, commanded=False):
        drive = self.terminal_command('vessel', commanded=commanded)
        epi = EPI_REST if commanded else self.c['adrenaline']
        concentrations = {'ne_vessel': drive, 'adrenaline': epi}
        a1 = self.organ_activation('a1_vessel', commanded=commanded) / self.organ_reference('a1_vessel')
        a2 = self.sig('a2b_vessel', concentrations, nodrug=commanded)
        b2 = self.organ_activation('b2_vessel', commanded=commanded)
        a20 = self.organ_reference('a2b_vessel')
        b20 = self.organ_reference('b2_vessel')
        return (1 - alpha1_weight + alpha1_weight * a1) * math.exp(
            COUPLING_SCALE * (VESSEL_A2B * (a2 - a20) - VESSEL_B2 * (b2 - b20)))

    def refresh_body_pools(self):
        self.c['ne_heart'] = self.terminal_command('heart')
        self.c['ne_vessel'] = self.terminal_command('vessel')
        self.c['ne_peripheral'] = (self.c['ne_heart'] + self.c['ne_vessel']) / 2
        self.c['ach_heart'] = self.body_vagal()
        self.c['ach_gland'] = self.c['ach_gut'] = self.body_vagal()

    def update_baroreflex(self, dt):
        """Backward Euler for the coupled pressure-error/reset feedback pair.

        Pressure depends on the new autonomic correction. Solving that dependence
        avoids the alternating overshoot of an explicit, stale-pressure update.
        Biological gains and time constants are unchanged.
        """
        old_error, old_adapt = self.baro_error, self.baro_adapt
        command_map = self.blood_pressure(commanded=True)
        response = dt / (BARO_TAU + dt)
        reset = dt / (BARO_RESET + dt)

        def residual(error):
            self.baro_error = error
            pressure = self.blood_pressure(self.inp)
            adapted = (1 - reset) * old_adapt + reset * (pressure - command_map)
            target = 30 * math.tanh((command_map + adapted - pressure) / 30)
            return error - ((1 - response) * old_error + response * target)

        try:
            accepted = bracketed_root(residual, min(-30., old_error), max(30., old_error))
        finally:
            self.baro_error = old_error
        self.baro_error = accepted
        pressure = self.blood_pressure(self.inp)
        self.baro_adapt = (1 - reset) * old_adapt + reset * (pressure - command_map)

    def update_respiration(self, dt, inp):
        """Implicit CO2/ventilation feedback, separate from amygdala acid sensing."""
        old_co2, old_vent = self.pco2, self.ventilation
        breath = dt / (1 / 60 + dt)
        gas = dt / (2 / 60 + dt)

        def ventilation(co2):
            return (1 - breath) * old_vent + breath * max(.1, 1 + .05 * (co2 - 40))

        if 'pco2_mmhg' in inp:
            target = inp['pco2_mmhg']
            if target <= 0 or not math.isfinite(target):
                raise ValueError('pco2_mmhg must be positive and finite')
            self.pco2 = (1 - gas) * old_co2 + gas * target
        else:
            production = 40 * self.receptor_factor('metabolism')
            retained = (1 - gas) * old_co2
            vent_floor = (1 - breath) * old_vent + breath * .1
            self.pco2 = bracketed_root(
                lambda co2: co2 - retained - gas * production / ventilation(co2),
                retained, retained + gas * production / vent_floor)
        self.ventilation = ventilation(self.pco2)
        self.brain_ph = inp.get('brain_ph', 6.1 + math.log10(24 / (.03 * self.pco2)))
        self.c['proton'] = 10**(6.5 - self.brain_ph)

    def body_update(self, dt, inp):
        def toward(key, target, tau):
            if target < 0 or not math.isfinite(target):
                raise ValueError(f'invalid body target {key}={target}')
            self.c[key] += (target - self.c[key]) * -math.expm1(-dt / tau)

        # Finite-delay pressure feedback. Central commands may change its set point;
        # peripheral drug effects are excluded from that command-only pressure.
        self.update_baroreflex(dt)

        command = self._sympathetic() / (self.cal['rest'].get('a1_symp') or self.eff('a1_symp'))
        toward('ach_ganglion', max(.0, 1 + .25 * (command - 1)), 1 / 60)
        gang = self.relative_signal('nic_ganglion')
        toward('adrenaline', EPI_REST * (1 + 4 * max(0., command - 1)) * gang, 2 / 60)
        self.refresh_body_pools()

        meal, fasting = max(0., inp.get('meal', 0.)), max(0., inp.get('fasting', 0.))
        toward('cck_gut', 1 + meal, .25)
        toward('cck_brain', 1 + .25 * max(0., inp.get('threat', 0.)), .25)
        toward('ht_gut', (1 + meal + max(0., inp.get('gut_irritant', 0.)))
               / (1 - self._block('sert', compartment='peripheral')), .25)
        toward('ha_gut', 1 + meal, .5)
        self.energy_store += ((1 + .1 * meal) / (1 + .1 * fasting) - self.energy_store) * -math.expm1(-dt / 12)
        toward('ghrelin', inp.get('ghrelin', (1 + fasting) / (1 + meal)), .5)
        toward('leptin', inp.get('leptin', self.energy_store), 6.)
        for hormone in ('thyroid', 'estrogen', 'androgen', 'progesterone'):
            toward(hormone, inp.get(hormone, 1.), 1.)
        toward('da_tida', max(0., 1 + .5 * self.dev('prlr')), .25)
        d2 = self.relative_signal('d2_pituitary')
        toward('prolactin', inp.get('prolactin', 2 / (1 + d2)), .5)
        pain = max(0., inp.get('pain', 0.))
        toward('substance_p', 1 + pain, .25)
        toward('enkephalin', 1 + .25 * pain + .25 * inp.get('touch', 0.), .5)
        toward('nociceptin', 1 + .25 * max(0., self.c['crh'] - 1), .5)
        toward('urocortin', 1 + .25 * max(0., self.c['crh'] - 1), 1.)
        toward('glycine', inp.get('glycine', 1.), .1)
        toward('glu_syn', max(0., 1 + inp.get('neural_input', 0.)), 1 / 60)
        toward('aea_peripheral', 1 / (1 + .25 * pain), .25)
        toward('ag2_peripheral', 1 + .25 * pain, .25)

        # CO2 drives respiratory control independently of the amygdala's ASIC gate.
        # An explicit PCO2 input is a clamp, not an inhaled-gas/dose conversion.
        self.update_respiration(dt, inp)

        # Read orthosteric agonist occupancy, not efficacy, for desensitization.
        for r in self.nic_available:
            occupancy = self.exogenous_agonist_occupancy(r)
            target = 1 - .9 * occupancy
            tau = NIC_DESENSITIZE if target < self.nic_available[r] else NIC_RECOVER
            self.nic_available[r] += (target - self.nic_available[r]) * -math.expm1(-dt / tau)
            amount_target = self.g.get('rx_' + r, 1.) * (1 + occupancy)
            self.amount[r] += (amount_target - self.amount[r]) * -math.expm1(-dt / NIC_UPREGULATE)

    def body_readouts(self):
        return {
            'vascular_resistance': self.vascular_tightness(self.vessel_alpha1_weight),
            'airway_resistance': math.exp(COUPLING_SCALE * .25 * (self.dev('m3_airway') - self.dev('b2_airway'))),
            'secretions': self.eff('m3_gland') / max(1e-12, self.cal['rest'].get('m3_gland', self.sig('m3_gland'))),
            'gastric_acid': self.eff('h2_stomach') / max(1e-12, self.cal['rest'].get('h2_stomach', self.sig('h2_stomach'))),
            'gut_motility': self.receptor_factor('motility'),
            'hunger': self.receptor_factor('hunger'),
            'pain': max(0., self.receptor_effect('pain')),
            'nausea': max(0., self.receptor_effect('nausea')),
            'metabolic_rate': self.receptor_factor('metabolism'),
            'reproductive_axis': self.receptor_factor('reproductive'),
            'motor_excitability': self.receptor_factor('motor'),
            'synaptic_excitation': self.eff('ampa') + self.eff('nmda_syn') + .25 * self.eff('mglur5'),
            'prolactin': self.c['prolactin'], 'adrenaline': self.c['adrenaline'],
            'peripheral_ne': self.c['ne_peripheral'], 'vagal': self.body_vagal(),
            'baroreflex_error': self.baro_error, 'baroreflex_adaptation': self.baro_adapt,
            'pco2_mmhg': self.pco2, 'brain_ph': self.brain_ph, 'ventilation': self.ventilation,
            'nic42_available': self.nic_available['nic42'],
        }
