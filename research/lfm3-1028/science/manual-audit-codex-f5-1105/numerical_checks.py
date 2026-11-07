import math,json
from pathlib import Path
from scipy.integrate import solve_ivp
p=Path('/private/tmp/littleguy-f5-manual-audit-1105');v={};R=8.314462618;NA=6.02214076e23;k=1.380649e-23;h=6.62607015e-34;e=1.602176634e-19
v['AgCl_CODATA_enthalpy_kJ_mol']=105.79-167.080+127.01
v['methane_radius_Vc_hardsphere_nm']=(98.7e-6/(16*math.pi*NA))**(1/3)*1e9
v['Cp_from_TV']=R+R*math.log(2)/math.log(298.15/248.44)
v['Cp_from_TP']=R*math.log(202.94/81.840)/math.log(298.15/248.44)
v['ideal_Pf_from_VT_kPa']=202.94/2*248.44/298.15
b=46/149.598;v['comet_perihelion_days']=math.sqrt(2*(1-b))*(1+2*b)/(3*math.pi)*365.256
b=57.909/149.598;v['comet_mean_radius_days']=math.sqrt(2*(1-b))*(1+2*b)/(3*math.pi)*365.256
u=14.9;a=math.radians(10);roots=[u*(2*math.cos(a)+sgn*math.sqrt(1-4*math.sin(a)**2))/3 for sgn in[-1,1]];v['neutron_speed_branches_km_s']=[math.sqrt(2*(u*u-z*z))for z in roots]
vi=2*R*298.15/(10*101325);vf=vi+.0002;tf=298.15-101325*.0002/(2*28.8);v['CO2_entropy_J_K']=2*28.8*math.log(tf/298.15)+2*R*math.log(vf/vi)
v['Bragg_delta_2theta_degrees']=2*(math.asin(154.433/155.6)-math.asin(154.051/155.6))*180/math.pi
v['Raoult_molar_mass_g_mol']=19/((500/78.11)*(53.3/51.5-1))
z=math.sqrt(2*9.1093837015e-31*.5*e)/(h/(2*math.pi))*1e-10;v['barrier_kappaL']=z;v['barrier_exact_T']=1/(1+4/(4*1.5*.5)*math.sinh(z)**2);v['barrier_exponential_only_T']=math.exp(-2*z)
v['hydrogen_supplied_potential_eV']=1.2*8.9875517923e9*e*1e-30/(5.29177210903e-11)**3;v['hydrogen_full_uniform_sphere_eV']=v['hydrogen_supplied_potential_eV']/3
beta=math.sqrt(2*k*298/(.0280134/NA));cdf=lambda w:math.erf(w/beta)-2/math.sqrt(math.pi)*w/beta*math.exp(-(w/beta)**2);v['N2_speed_fraction_200_300']=cdf(300)-cdf(200)
v['NOF_partition']=math.sqrt(math.pi)*(k*298.15/(h*2.99792458e10))**1.5/math.sqrt(3.1752*.3951*.3505)
v['bond_energy_J']=1.5*(500*(.133e-10)**2+950*(.062e-10)**2)
v['mixing_volume_cm3']=3.75*17+(3.75*.55/.45)*57.5-3.75*18.015/.997-(3.75*.55/.45)*46.07/.7893
v['argon_P_bar']=.08314462618*426/(1.31-.032)-1.355/1.31**2
# Independent model comparison for specified rocket idealizations; never executes generated reply code.
def flight(variable_g=False,variable_rho=False,drag=True):
 def rhs(t,y):
  alt,speed=y;mass=max(1e4,1e5-900*t);thrust=3.6e6 if t<100 else 0
  grav=9.8*(6371000/(6371000+alt))**2 if variable_g else 9.8
  rho=10**(.11-.05*alt/1000)if variable_rho else 1.29
  force=.5*.2*rho*math.pi*.2**2*speed*abs(speed)if drag else 0
  return[speed,(thrust-force)/mass-grav]
 def apex(t,y):return y[1]
 apex.direction=-1;apex.terminal=True
 a=solve_ivp(rhs,[0,100],[0,0],rtol=1e-9,atol=1e-9,max_step=.2)
 b=solve_ivp(rhs,[100,30000],a.y[:,-1],rtol=1e-9,atol=1e-9,max_step=2,events=apex)
 return{'burnout_height_m':float(a.y[0,-1]),'burnout_speed_m_s':float(a.y[1,-1]),'max_height_km':float(b.y[0,-1]/1000),'apex_reached':bool(len(b.t_events[0]))}
v['rocket_assumptions']='Fd=0.5*Cw*rho*pi*r^2*v^2; sea-levelrho1.29kg/m3; EarthR6371km; g0=9.8; all supplied fuel burned in100s.'
v['rocket_no_drag_constant_g']=flight(drag=False);v['rocket_constant_rho_g']=flight();v['rocket_constant_rho_variable_g']=flight(True);v['rocket_variable_rho_g']=flight(True,True)
(p/'numerical-checks.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
