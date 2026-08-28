from matplotlib import pyplot as plt
import numpy as np
import time
import os
import argparse
from mpi4py import MPI
import matplotlib.ticker as ticker
import cmocean
import torch
from nn_model import MoistExchangesNN

from NonEqEuler_NewConst import _NonEqEuler2D, pot_temp, chem_pots, eta_k_to_eta_d

def __get_thermodynamic_quantities(density, entropy, qv, ql, qi):
    _Rd = 287.0
    _Rv = 461.0
    _cpd = 1004.0
    _cvd = _cpd - _Rd
    _cpv = 1885.0
    _cvv = _cpv - _Rv
    _cl = 4186.0
    _ci = 2106.0
    _logRd = np.log(_Rd)
    _T0 = 273.15
    _p0 = 611.2 # ??? 1.0e+5??
    _psat0 = 611.2
    _rho0 = _p0 / (_Rv * _T0)
    _logT0 = np.log(_T0)
    Lv0_ = 2.5e6
    Ls0_ = 2.834e6
    Lf0_ = Ls0_ - Lv0_
    _Lv0 = Lv0_ - (_cpv - _cl) * _T0
    _Ls0 = Ls0_ - (_cpv - _ci) * _T0
    _Lf0 = _Ls0 - _Lv0
    _c0 = _cpv + (_Ls0 / _T0) - _cvv * _logT0 + _Rv * np.log(_rho0)
    _c1 = _cl + (_Lf0 / _T0) - _cl * _logT0
    _c2 = _ci - _ci * _logT0

    qw = qv + ql + qi
    qd = 1 - qw

    R = qv * _Rv + qd * _Rd
    cv = qd * _cvd + qv * _cvv + ql * _cl + qi * _ci

    logqv = np.log(qv)
    logqd = np.log(qd)
    logdensity = np.log(density)

    cvlogT = entropy + R * logdensity + qd * _Rd * (logqd + _logRd) + qv * _Rv * logqv
    cvlogT += -qv * _c0 - ql * _c1 - qi * _c2
    logT = (1 / cv) * cvlogT
    T = np.exp(logT)

    p = density * R * T

    specific_ie = cv * T + qv * _Ls0 + ql * _Lf0
    enthalpy = specific_ie + p / density
    ie = density * specific_ie

    dlogTdqv = (1 / cv) * (_Rv * logdensity + _Rv * logqv + _Rv - _c0)
    dlogTdqv += -(1 / cv) * logT * _cvv
    dTdqv = dlogTdqv * T

    dlogTdql = (1 / cv) * (-_c1)
    dlogTdql += -(1 / cv) * logT * _cl
    dTdql = dlogTdql * T

    dlogTdqi = (1 / cv) * (-_c2)
    dlogTdqi += -(1 / cv) * logT * _ci
    dTdqi = dlogTdqi * T

    #dlogTdqd = (1 / cv) * (_Rd * logdensity + _Rd * (logqd + np.log(_Rd)) + _Rd)
    #dlogTdqd += -(1 / cv) * logT * _cvd
    #dTdqd = T * dlogTdqd

    # these are just the Gibbs functions
    #mu_d = cv * dTdqd + _cvd * T
    mu_v = cv * dTdqv + _cvv * T + _Ls0
    mu_l = cv * dTdql + _cl * T + _Lf0
    mu_i = cv * dTdqi + _ci * T

    return mu_v, mu_l, mu_i, T#, mu_d

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, help='Number of cells')
parser.add_argument('--o', type=int, help='Polynomial order')
parser.add_argument('--nproc', type=int, help='Number of procs', default=1)
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

xlim = 10_000
zlim = 10_000

nz = args.n
nproc = args.nproc
run_model = (not args.plot)  # whether to run model - set false to just plot previous run
nx = nz
cfl = 0.5
g = 9.81
poly_order = args.o
#a = 0.5
a = 0
upwind = True
non_equilibrium_thermo = True

exp_name_short = 'bf02-repo4'
if a == 0:
    exp_name_short = exp_name_short + '-energy-conserving'
experiment_name = f'{exp_name_short}-nx-{nx}-nz-{nz}-p{poly_order}'
data_dir = os.path.join('data', experiment_name)
plot_dir = os.path.join('plots', experiment_name)

if rank == 0:
    print(f"---------- Ice bubble with nx={nx}, nz={nz}, cfl={cfl}")
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    if not os.path.exists(data_dir): os.makedirs(data_dir)

comm.barrier()
#

zmap = lambda x, z: z * zlim
xmap = lambda x, z: xlim * (x - 0.5)

v_power_list = []
l_power_list = []
i_power_list = []
e_power_list = []
v_water_list = []
l_water_list = []
i_water_list = []
entropy_list = []

def update_monotone(udq1, udq2, hdt, inc):
    epsilon = 1.0e-14
    use_both = np.logical_and(udq1 + hdt * inc > epsilon, udq2 - hdt * inc > epsilon)
    not_both = np.logical_not(use_both)
    alpha_1 = -udq1 / (hdt * inc)
    alpha_1 = np.maximum(alpha_1, 0.0)
    alpha_1 = np.minimum(alpha_1, 0.99999999)
    alpha_2 = +udq2 / (hdt * inc)
    alpha_2 = np.maximum(alpha_2, 0.0)
    alpha_2 = np.minimum(alpha_2, 0.99999999)
    alpha = np.minimum(alpha_1, alpha_2)
    use = np.ones(inc.shape) * use_both + alpha * not_both
    return use

def forcing_function(solver, state, dstatedt, state_0, hdt):
    u, w, h, s, qv, ql, qi = solver.get_vars(state)
    dudt, dwdt, dhdt, dsdt, dqvdt, dqldt, dqidt = solver.get_vars(dstatedt)
    _, _, _, _, qv0, ql0, qi0 = solver.get_vars(state_0)

    # add heating terms in dsdt
    #T, mu_v, mu_l, mu_i = chem_pots(h, s_d, qv, ql, qi)
    mu_v, mu_l, mu_i, T = __get_thermodynamic_quantities(h, s, qv, ql, qi)

    if non_equilibrium_thermo:
        vp = 0.0
        lp = 0.0
        ip = 0.0
        ep = 0.0
        # parse therodynamic state to pytorch array
        #nn_in = torch.from_numpy(np.array([qv.flatten(), \
        #                                   ql.flatten(), \
        #                                   qi.flatten(), \
        #                                   s.flatten(), \
        #                                   h.flatten()]).transpose())
        nn_in = torch.from_numpy(np.array([qv.flatten(), \
                                           ql.flatten(), \
                                           qi.flatten(), \
                                           s.flatten(), \
                                           h.flatten()]).transpose().astype(np.float32))
        # evaluate the nn and parse back as numpy array
        mp_incs = model(nn_in).detach().numpy().reshape([T.shape[0],T.shape[1],T.shape[2],T.shape[3],3])
        mp_incs[:,:,:,:,0] = (qv + ql) * mp_incs[:,:,:,:,0]
        mp_incs[:,:,:,:,1] = (qv + qi) * mp_incs[:,:,:,:,1]
        mp_incs[:,:,:,:,2] = (ql + qi) * mp_incs[:,:,:,:,2]
        # solution increments from transport for checking monotonicity
        epsilon = 1.0e-14
        # vapor to liquid exchange
        u_dqv = qv0 + hdt * dqvdt
        u_dql = ql0 + hdt * dqldt
        u_dqi = qi0 + hdt * dqidt
        inc    = mp_incs[:,:,:,:,0] * h * (mu_v - mu_l)
        inc_s  = inc * (mu_v - mu_l) / T
        use    = np.logical_and(u_dqv + hdt * inc > epsilon, u_dql - hdt * inc > epsilon)
        #use    = update_monotone(u_dqv, u_dql, hdt, inc)
        dqvdt += use * inc
        dqldt -= use * inc
        dsdt  -= use * inc_s
        vp += solver.integrate(use*h*h*mp_incs[:,:,:,:,0]*mu_v*(mu_v - mu_l))
        lp -= solver.integrate(use*h*h*mp_incs[:,:,:,:,0]*mu_l*(mu_v - mu_l))
        ep -= solver.integrate(use*h*h*mp_incs[:,:,:,:,0]*(mu_v - mu_l)*(mu_v - mu_l))
        # vapor to ice exchange
        u_dqv = qv0 + hdt * dqvdt
        u_dql = ql0 + hdt * dqldt
        u_dqi = qi0 + hdt * dqidt
        inc    = mp_incs[:,:,:,:,1] * h * (mu_v - mu_i)
        inc_s  = inc * (mu_v - mu_i) / T
        use    = np.logical_and(u_dqv + hdt * inc > epsilon, u_dqi - hdt * inc > epsilon)
        #use    = update_monotone(u_dqv, u_dqi, hdt, inc)
        dqvdt += use * inc
        dqidt -= use * inc
        dsdt  -= use * inc_s
        vp += solver.integrate(use*h*h*mp_incs[:,:,:,:,1]*mu_v*(mu_v - mu_i))
        ip -= solver.integrate(use*h*h*mp_incs[:,:,:,:,1]*mu_i*(mu_v - mu_i))
        ep -= solver.integrate(use*h*h*mp_incs[:,:,:,:,1]*(mu_v - mu_i)*(mu_v - mu_i))
        # liquid to ice exchange
        u_dqv = qv0 + hdt * dqvdt
        u_dql = ql0 + hdt * dqldt
        u_dqi = qi0 + hdt * dqidt
        inc    = mp_incs[:,:,:,:,2] * h * (mu_l - mu_i)
        inc_s  = inc * (mu_l - mu_i) / T
        use    = np.logical_and(u_dql + hdt * inc > epsilon, u_dqi - hdt * inc > epsilon)
        #use    = update_monotone(u_dql, u_dqi, hdt, inc)
        dqldt += use * inc
        dqidt -= use * inc
        dsdt  -= use * inc_s
        lp += solver.integrate(use*h*h*mp_incs[:,:,:,:,2]*mu_l*(mu_l - mu_i))
        ip -= solver.integrate(use*h*h*mp_incs[:,:,:,:,2]*mu_i*(mu_l - mu_i))
        ep -= solver.integrate(use*h*h*mp_incs[:,:,:,:,2]*(mu_l - mu_i)*(mu_l - mu_i))
        v_power_list.append(vp)
        l_power_list.append(lp)
        i_power_list.append(ip)
        e_power_list.append(ep)
        v_water_list.append(solver.integrate(h*qv))
        l_water_list.append(solver.integrate(h*ql))
        i_water_list.append(solver.integrate(h*qi))
        entropy_list.append(solver.integrate(h*s))
    else:
        # simple scheme - always moving towards equilibrium
        qw = qv + ql + qi
        qv_eq, ql_eq, qi_eq = solver.solve_fractions_from_entropy(h, qw, s)
        time_scale = 4.0
        dqvdt_microphysics = (qv_eq - qv) / time_scale
        dqldt_microphysics = (ql_eq - ql) / time_scale
        dqidt_microphysics = (qi_eq - qi) / time_scale
        dsdt_microphysics = -(mu_v * dqvdt_microphysics + mu_l * dqldt_microphysics + mu_i * dqidt_microphysics) / T
        dsdt  += dsdt_microphysics
        dqvdt += dqvdt_microphysics
        dqldt += dqldt_microphysics
        dqidt += dqidt_microphysics

def initial_condition(xs, ys, solver, pert):
    u = 0 * ys
    v = 0 * ys

    dry_theta = 300
    dexdy = -g / (solver.cpd * dry_theta)
    ex = 1 + dexdy * ys
    p = 1_00_000.0 * ex ** (solver.cpd / solver.Rd)
    density = p / (solver.Rd * ex * dry_theta)

    qw = solver.rh_to_qw(0.95, p, density)
    qd = 1 - qw

    R = solver.Rd * qd + solver.Rv * qw
    T = p / (R * density)

    assert (qw <= solver.saturation_fraction(T, density)).all()

    rad_max = 2_000
    rad = np.sqrt(xs ** 2 + (ys - 1.0 * rad_max) ** 2)
    mask = rad < rad_max
    density -= mask * (pert * density / 300) * (np.cos(np.pi * (rad / rad_max) / 2) ** 2)

    T = p / (R * density)
    assert (qw <= solver.saturation_fraction(T, density)).all()

    s = qd * solver.entropy_air(T, qd, density)
    s += qw * solver.entropy_vapour(T, qw, density)

    qv, ql, qi = solver.solve_fractions_from_entropy(density, qw, s)
    print('qw min-max:', qw.min(), qw.max())
    print('T min-max:', T.min() - 273, T.max() - 273)
    print('s min-max:', s.min(), s.max(), s.sum()/len(s.flatten()))
    print('Density min-max:', density.min(), density.max())
    print('Pressure min-max:', p.min(), p.max())
    print('qv/qw min-max:', (qv/qw).min(), (qv/qw).max())
    print('all vapour mean:', (qv == qw).mean())
    print('ql/qw min-max:', (ql/qw).min(), (ql/qw).max())
    print('qi/qw min-max:', (qi/qw).min(), (qi/qw).max(), '\n')

    return u, v, density, s, qw, qv, ql, qi

tends = np.array([0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0])

conservation_data_fp = os.path.join(data_dir, 'conservation_data.npy')
time_list = []
energy_list = []
entropy_var_list = []
water_var_list = []

if non_equilibrium_thermo:
    model_path = '/g/data/dp9/dl9118/lfric_ral_training/repo4/src/results/model_min_loss_17.pt'
    #model = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
    #model.eval()
    checkpoint = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
    model = MoistExchangesNN()
    model.load_state_dict(checkpoint['model_state_dict'])

if run_model:
    solver = _NonEqEuler2D(
        xmap, zmap, poly_order, nx,
        g=g, cfl=cfl, a=a, nz=nz, upwind=upwind, nprocx=nproc,
        forcing=forcing_function
    )
    u, v, density, s, qw, qv, ql, qi = initial_condition(solver.xs, solver.zs, solver, pert=2.0)
    solver.set_initial_condition(u, v, density, s, qv, ql, qi)
    for i, tend in enumerate(tends):
        t0 = time.time()
        while solver.time < tend:
            time_list.append(solver.time)
            energy_list.append(solver.energy())
            entropy_var_list.append(solver.integrate(solver.h * solver.s ** 2))
            water_var_list.append(solver.integrate(solver.h * solver.qv ** 2))

            dt = min(solver.get_dt(), tend - solver.time)
            solver.time_step(dt=dt)
        t1 = time.time()

        if rank == 0:
            print("Simulation time (unit less):", solver.time)
            print("Wall time:", time.time() - t0, '\n')

        solver.save(solver.get_filepath(data_dir, exp_name_short))

    if rank == 0:
        conservation_data = np.zeros((12, len(time_list)))
        conservation_data[0, :] = np.array(time_list)
        conservation_data[1, :] = np.array(energy_list)
        conservation_data[2, :] = np.array(entropy_var_list)
        conservation_data[3, :] = np.array(water_var_list)
        conservation_data[4, :] = np.array(v_power_list)[::4]
        conservation_data[5, :] = np.array(l_power_list)[::4]
        conservation_data[6, :] = np.array(i_power_list)[::4]
        conservation_data[7, :] = np.array(e_power_list)[::4]
        conservation_data[8, :] = np.array(v_water_list)[::4]
        conservation_data[9, :] = np.array(l_water_list)[::4]
        conservation_data[10,:] = np.array(i_water_list)[::4]
        conservation_data[11,:] = np.array(entropy_list)[::4]
        np.save(conservation_data_fp, conservation_data)

        print('Energy error:', (energy_list[-1] - energy_list[0]) / energy_list[0])

    print('Time of first limit:', solver.first_water_limit_time)


# plotting
elif rank == 0:
    plt.rcParams['font.size'] = '12'

    conservation_data = np.load(conservation_data_fp)
    time_list = conservation_data[0, :]
    mask = time_list <= np.inf
    energy_list = conservation_data[1, :][mask]
    entropy_var_list = conservation_data[2, :][mask]
    water_var_list = conservation_data[3, :][mask]
    v_power = conservation_data[4, :][mask]
    l_power = conservation_data[5, :][mask]
    i_power = conservation_data[6, :][mask]
    e_power = conservation_data[7, :][mask]
    v_water = conservation_data[8, :][mask]
    l_water = conservation_data[9, :][mask]
    i_water = conservation_data[10,:][mask]
    entropy = conservation_data[11,:][mask]
    time_list = time_list[mask]

    e_diff = abs(np.diff(energy_list))
    print('Time max energy growth:', time_list[np.argmax(e_diff) + 1])

    energy_list = (energy_list - energy_list[0]) / energy_list[0]
    entropy_var_list = (entropy_var_list - entropy_var_list[0]) / entropy_var_list[0]
    water_var_list = (water_var_list - water_var_list[0]) / water_var_list[0]

    print('Energy error:', energy_list[-1])
    print('Entropy var error:', entropy_var_list[-1])
    print('Water var error:', water_var_list[-1])

    plt.figure()
    plt.plot(time_list, energy_list, label='Energy')
    plt.plot(time_list, entropy_var_list, label='Entropy variance')
    plt.plot(time_list, water_var_list, label='Water variance')
    plt.grid()
    plt.legend()
    plt.ylabel('Relative error')
    plt.xlabel('Time (s)')
    plt.yscale('symlog', linthresh=1e-15)
    fp = os.path.join(plot_dir, f'conservation_{exp_name_short}')
    plt.savefig(fp, bbox_inches="tight")

    plt.figure()
    plt.plot(time_list[1:], v_power[1:], label='Vapor')
    plt.plot(time_list[1:], l_power[1:], label='Liquid')
    plt.plot(time_list[1:], i_power[1:], label='Ice')
    plt.plot(time_list[1:], e_power[1:], label='Entropy')
    plt.plot(time_list[1:], v_power[1:] + l_power[1:] + i_power[1:] + e_power[1:], label='Total')
    plt.grid()
    plt.legend()
    plt.ylabel('Watts (J/s)')
    plt.xlabel('Time (s)')
    plt.title('Power')
    plt.yscale('symlog', linthresh=1e-15)
    fp = os.path.join(plot_dir, f'power_{exp_name_short}')
    plt.savefig(fp, bbox_inches="tight")

    plt.figure()
    t_water = v_water + l_water + i_water
    plt.plot(time_list[1:], v_water[1:], label='Vapor')
    plt.plot(time_list[1:], l_water[1:], label='Liquid')
    plt.plot(time_list[1:], i_water[1:], label='Ice')
    #plt.plot(time_list[1:], (t_water[1:]-t_water[0])/t_water[0], label='Total water change (normalised)')
    plt.plot(time_list[1:], entropy[1:]-entropy[0], label='Entropy change')
    plt.grid()
    plt.legend()
    plt.ylabel('Mass (kg)')
    plt.xlabel('Time (s)')
    plt.title('Water')
    plt.yscale('symlog', linthresh=1e-15)
    fp = os.path.join(plot_dir, f'water_{exp_name_short}')
    plt.savefig(fp, bbox_inches="tight")

    solver_plot = _NonEqEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=0.5, a=a, nz=nz, upwind=upwind, nprocx=1)
    _, _, _, s0, qw0, qv0, ql0, qi0 = initial_condition(solver_plot.xs, solver_plot.zs, solver_plot, pert=0.0)


    def fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)


    plot_func_entropy = lambda s: s.project_H1(s.s - s0)
    plot_func_density = lambda s: s.project_H1(s.h)
    plot_func_water = lambda s: s.project_H1(s.qv + s.ql + s.qi)
    plot_func_vapour = lambda s: s.project_H1(s.qv)
    plot_func_liquid = lambda s: s.project_H1(s.ql)
    plot_func_ice = lambda s: s.project_H1(s.qi)

    fig_list = [plt.subplots(2, 3, sharex=True, sharey=True, figsize=(7.4, 4.8)) for _ in range(6)]

    pfunc_list = [
        plot_func_entropy, plot_func_density,
        plot_func_water, plot_func_vapour, plot_func_liquid, plot_func_ice
    ]

    labels = ["entropy", "density", "water", "vapour", "liquid", "ice"]

    energy = []
    tends = np.array([0.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    for i, tend in enumerate(tends):
        filepaths = [solver_plot.get_filepath(data_dir, exp_name_short, proc=i, nprocx=nproc, time=tend) for i in range(nproc)]
        solver_plot.load(filepaths)
        energy.append(solver_plot.integrate(solver_plot.energy()))

        for (fig, axs), plot_fun, label in zip(fig_list, pfunc_list, labels):
            ax = axs[i // 3][i % 3]
            ax.tick_params(labelsize=8)
            #ax.set_box_aspect(0.33)

            if label == 'ice':
                # levels = np.linspace(0.0, 7e-3, 1000)
                levels = 1000
                cmap = cmap = cmocean.cm.ice
            elif label == 'entropy':
                # levels = np.linspace(-30, 70, 1000)
                levels = 1000
                #cmap = cmap = cmocean.cm.thermal
                cmap = 'nipy_spectral'
            else:
                levels = 1000
                cmap = 'nipy_spectral'

            im = solver_plot.plot_solution(ax, dim=2, plot_func=plot_fun, levels=levels, cmap=cmap)
            # if label == 'entropy':
            #     cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt), label='Entropy (K)')
            # elif label == 'density':
            #     cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt), label='Density ($\text{kg m}^{-3}$)')
            # else:
            #     cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt), label=f'{label.capitalize() mass fraction'})
            #cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt))
            cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt), orientation='horizontal')
            ticks = np.linspace(im.norm.vmin, im.norm.vmax, 3)
            cbar.set_ticks(ticks)
            cbar.ax.tick_params(labelsize=8)

            if (i // 3) == 1:
                ax.set_xlabel('x (m)', fontsize='xx-small')
            if (i % 3) == 0:
                ax.set_ylabel('z (m)', fontsize='xx-small')
            # fig.tight_layout(w_pad=1.0, h_pad=1.0)
            fig.tight_layout()

    for (fig, ax), label in zip(fig_list, labels):
        plot_name = f'{label}_{exp_name_short}'
        fp = solver_plot.get_filepath(plot_dir, plot_name, ext='png')
        fig.savefig(fp, bbox_inches="tight")

    tends = np.array([0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0])
    for tend in tends:
        filepaths = [solver_plot.get_filepath(data_dir, exp_name_short, proc=i, nprocx=nproc, time=tend) for i in range(nproc)]
        solver_plot.load(filepaths)
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.4, 4.8))
        # entropy
        ax = axs[0][0]
        ax.set_title('Entropy perturbation',fontsize=10)
        ax.tick_params(labelsize=8)
        im = solver_plot.plot_solution(ax, dim=2, plot_func=plot_func_entropy, levels=1000, cmap='nipy_spectral')
        cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt), orientation='vertical')
        ticks = np.linspace(im.norm.vmin, im.norm.vmax, 3)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=8)

        # vapour
        ax = axs[0][1]
        ax.tick_params(labelsize=8)
        ax.set_title('Vapour',fontsize=10)
        im = solver_plot.plot_solution(ax, dim=2, plot_func=plot_func_vapour, levels=1000, cmap='nipy_spectral')
        cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt), orientation='vertical')
        ticks = np.linspace(im.norm.vmin, im.norm.vmax, 3)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=8)

        # liquid
        ax = axs[1][0]
        ax.set_title('Liquid',fontsize=10)
        ax.tick_params(labelsize=8)
        im = solver_plot.plot_solution(ax, dim=2, plot_func=plot_func_liquid, levels=1000, cmap='nipy_spectral')
        cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt), orientation='vertical')
        ticks = np.linspace(im.norm.vmin, im.norm.vmax, 3)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=8)

        # ice
        ax = axs[1][1]
        ax.set_title('Ice',fontsize=10)
        ax.tick_params(labelsize=8)
        im = solver_plot.plot_solution(ax, dim=2, plot_func=plot_func_ice, levels=1000, cmap='nipy_spectral')
        cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt), orientation='vertical')
        ticks = np.linspace(im.norm.vmin, im.norm.vmax, 3)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=8)

        plot_name = f'{exp_name_short}_{tend}'
        fp = solver_plot.get_filepath(plot_dir, plot_name, ext='png')
        fig.savefig(fp, bbox_inches="tight")
