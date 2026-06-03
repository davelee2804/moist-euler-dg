from matplotlib import pyplot as plt
from moist_euler_dg.non_equilibrium_euler_2D import NonEqEuler2D
import numpy as np
import time
import os
import argparse
from mpi4py import MPI
import matplotlib.ticker as ticker
import cmocean
import torch
from nn_model import MoistExchangesNN

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
a = 0.5
upwind = True

non_equilibrium_thermo_2phase = False
non_equilibrium_thermo_3phase = True

exp_name_short = 'ice-bubble-03'
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

def chem_pots(rho, eta, q_v, q_l, q_i):
    p_0d     = 1.0e+5
    p_0sat   = 611.2
    p_0v     = p_0sat
    R_d      = 287.5 # LFRic
    R_v      = 461.51 # LFRic
    T_0      = 273.15
    alpha_0d = R_d*T_0/p_0d
    alpha_0v = R_v*T_0/p_0v
    alpha_l  = 0.001
    alpha_i  = 0.0011
    c_pd     = 1005.0 # LFRic
    c_pv     = 1885.0
    c_vd     = c_pd - R_d
    c_vv     = c_pv - R_v
    c_i      = 2106.0
    c_l      = 4186.0
    L_0s     = 2.834e+6
    L_0v     = 2.5e+6
    L_0f     = L_0s - L_0v
    L_00s    = L_0s - (c_pv - c_i)*T_0 + alpha_i*p_0v
    L_00v    = L_0v - (c_pv - c_l)*T_0 + alpha_l*p_0v
    L_00f    = L_00s - L_00v
    eta_0    = 0.0

    q_d = 1.0 - q_v - q_l - q_i

    # Derivatives of the internal energy with respect to the
    # mass fractions, with the internal energy given as:
    # Eldred et al., QJRMS (2022) eqn. 53
    #c_p = q_d*c_pd + q_v*c_pv + q_l*c_l + q_i*c_i
    #c_v = q_d*c_vd + q_v*c_vv + q_l*c_l + q_i*c_i

    #a = np.exp((eta - eta_0)/c_v)
    #b = np.power(alpha_0d*q_d*rho,R_d*q_d/c_v)
    #c = np.power(alpha_0v*q_v*rho,R_v*q_v/c_v)

    #T = T_0*a*b*c

    # scale by c_vj for phase j
    #da_dq = a*(eta_0 - eta)/(c_v*c_v)

    # d(a^f(x))/dx = a^f(x) . log(a) . df(x)/dx
    # scale by c_vj for phase j
    #db_dq = -b*np.log(alpha_0d*q_d*rho)*R_d*q_d/(c_v*c_v)

    # scale by c_vj for phase j = l,i
    #dc_dq = -c*np.log(alpha_0v*q_v*rho)*R_v*q_v/(c_v*c_v)

    # d(a^f(x))/dx = a(x)^f(x) [ log(a) . df(x)/dx + f(x) da(x)/dx / a(x) ]
    #dc_dqv = c*(np.log(alpha_0v*q_v*rho)*(c_v*R_v - c_vv*R_v*q_v)/(c_v*c_v) + R_v*q_v/(c_v*q_v))

    #tmp1 = T - T_0
    #tmp2 = T_0*c_v*da_dq*b*c
    #tmp3 = T_0*c_v*a*db_dq*c

    #mu_v = c_vv*tmp1 + c_vv*(tmp2 + tmp3) + T_0*c_v*    a*b*dc_dqv - R_v*T_0 + (L_00v + L_00f)
    #mu_l = c_l *tmp1 + c_l *(tmp2 + tmp3) + T_0*c_v*c_l*a*b*dc_dq  + L_00f
    #mu_i = c_i *tmp1 + c_i *(tmp2 + tmp3) + T_0*c_v*c_i*a*b*dc_dq

    #mu_v = solver.gibbs_vapour(T, q_v, rho)
    #mu_l = solver.gibbs_liquid(T)
    #mu_i = solver.gibbs_ice(T)

    #return T, mu_v, mu_l, mu_i

    # Derivatives of the internal energy with respect to the 
    # mass fractions, with the internal energy given as:
    # Eldred et al., QJRMS (2022) eqn. 53
    c_v = q_d * c_vd + q_v * c_vv + q_l * c_l + q_i * c_i
    c_v_inv = 1.0 / c_v

    _a = np.exp((eta - eta_0) * c_v_inv)
    _b = np.power(alpha_0d * q_d * rho, R_d * q_d * c_v_inv)
    _c = np.power(alpha_0v * q_v * rho, R_v * q_v * c_v_inv)
    T = T_0 * _a * _b * _c

    # d(a^f(x))/dx = a^f(x) . log(a) . df(x)/dx
    _ad = q_d * R_d * np.log(q_d * rho * alpha_0d)
    _av = q_v * R_v * np.log(q_v * rho * alpha_0v)
    _tmp = T - c_v_inv * T * (_ad + _av) - T_0

    mu_v = c_vv * _tmp + T * R_v * np.log(q_v * rho * alpha_0v) + \
           c_v * T_0 * _a * _b * rho * alpha_0v * np.power(alpha_0v * q_v * rho, R_v * q_v * c_v_inv - 1.0) - \
           R_v * T_0 + L_00f + L_00s
    mu_l = c_l * _tmp + L_00f
    mu_i = c_i * _tmp

    return T, mu_v, mu_l, mu_i

def forcing_function(solver, state, dstatedt):
    u, w, h, s, qv, ql, qi = solver.get_vars(state)
    dudt, dwdt, dhdt, dsdt, dqvdt, dqldt, dqidt = solver.get_vars(dstatedt)

    # add heating terms in dsdt
    T, mu_v, mu_l, mu_i = chem_pots(h, s, qv, ql, qi)

    if non_equilibrium_thermo_2phase:
        # parse therodynamic state to pytorch array
        scale = 1.0e+6
        hdt = 0.5 * solver.get_dt()
        nn_in = torch.from_numpy(np.array([mu_v.flatten()/scale, \
                                           mu_l.flatten()/scale, \
                                           T.flatten(), \
                                           h.flatten()]).transpose()).float()
        # evaluate the nn and parse back as numpy array
        mp_incs = model(nn_in).detach().numpy().reshape(T.shape)
        # check monotonicity
        inc = hdt * mp_incs * (mu_v - mu_l) / (scale * scale)
        use = ql > inc
        # apply incrememnts where not breaking monotonicity
        s  -= use * inc * (mu_v - mu_l) / T
        qv += use * inc
        ql -= use * inc
    elif non_equilibrium_thermo_3phase:
        # parse therodynamic state to pytorch array
        scale = 1.0e+6
        hdt = 0.5 * solver.get_dt()
        nn_in = torch.from_numpy(np.array([mu_v.flatten()/scale, \
                                           mu_l.flatten()/scale, \
                                           mu_i.flatten()/scale, \
                                           T.flatten(), \
                                           h.flatten()]).transpose()).float()
        # evaluate the nn and parse back as numpy array
        mp_incs = model(nn_in).detach().numpy().reshape([T.shape[0],T.shape[1],T.shape[2],T.shape[3],3])
        # check monotonicity
        inc_v = hdt * (mp_incs[:,:,:,:,0] * (mu_v - mu_l) + mp_incs[:,:,:,:,1] * (mu_v - mu_i)) / scale / scale
        inc_l = hdt * (mp_incs[:,:,:,:,0] * (mu_l - mu_v) + mp_incs[:,:,:,:,2] * (mu_l - mu_i)) / scale / scale
        inc_i = hdt * (mp_incs[:,:,:,:,1] * (mu_i - mu_v) + mp_incs[:,:,:,:,2] * (mu_i - mu_l)) / scale / scale
        u_dqv = qv + hdt * dqvdt
        u_dql = ql + hdt * dqldt
        u_dqi = qi + hdt * dqidt
        use_v = u_dqv > inc_v
        use_l = u_dql > inc_l
        use_i = u_dqi > inc_i
        use_li = np.logical_and(use_l, use_i)
        use    = np.logical_and(use_v, use_li)
        # apply incrememnts where not breaking monotonicity
        dsdt  -= use * ( (mu_v - mu_l) * (mu_v - mu_l) * mp_incs[:,:,:,:,0] + \
                         (mu_v - mu_i) * (mu_v - mu_i) * mp_incs[:,:,:,:,1] + \
                         (mu_l - mu_i) * (mu_l - mu_i) * mp_incs[:,:,:,:,2] ) / scale / scale / T
        dqvdt += use * inc_v
        dqldt += use * inc_l
        dqidt += use * inc_i
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
    #  0.3410208713540216 0.10594892674155956 0.6589791286459784
    # print('qw min-max:', qw.min(), qw.max())
    # print('T min-max:', T.min() - 273, T.max() - 273)
    # print('Density min-max:', density.min(), density.max())
    # print('Pressure min-max:', p.min(), p.max())
    # print('qv/qw min-max:', (qv/qw).min(), (qv/qw).max())
    # print('all vapour mean:', (qv == qw).mean())
    # print('ql/qw min-max:', (ql/qw).min(), (ql/qw).max())
    # print('qi/qw min-max:', (qi/qw).min(), (qi/qw).max(), '\n')

    return u, v, density, s, qw, qv, ql, qi


run_time = 600

tends = np.array([0.0, 200.0, 400.0, 600.0])
# tends = np.array([0.0, 200.0, 400.0, 479.0])

conservation_data_fp = os.path.join(data_dir, 'conservation_data.npy')
time_list = []
energy_list = []
entropy_var_list = []
water_var_list = []

if non_equilibrium_thermo_2phase:
    model_path = '/g/data/dp9/dl9118/lfric_ral_training/runs/two_phase/prev_07/model_min_loss.pt'
elif  non_equilibrium_thermo_3phase:
    model_path = '/g/data/dp9/dl9118/lfric_ral_training/runs/nAdv_nEta_wBatch/model_min_loss.pt'

if non_equilibrium_thermo_2phase or non_equilibrium_thermo_3phase:
    model = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
    model.eval()

if run_model:
    solver = NonEqEuler2D(
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
        conservation_data = np.zeros((4, len(time_list)))
        conservation_data[0, :] = np.array(time_list)
        conservation_data[1, :] = np.array(energy_list)
        conservation_data[2, :] = np.array(entropy_var_list)
        conservation_data[3, :] = np.array(water_var_list)
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

    solver_plot = NonEqEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=0.5, a=a, nz=nz, upwind=upwind, nprocx=1)
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

    fig_list = [plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.4, 4.8)) for _ in range(6)]

    pfunc_list = [
        plot_func_entropy, plot_func_density,
        plot_func_water, plot_func_vapour, plot_func_liquid, plot_func_ice
    ]

    labels = ["entropy", "density", "water", "vapour", "liquid", "ice"]

    energy = []
    for i, tend in enumerate(tends):
        filepaths = [solver_plot.get_filepath(data_dir, exp_name_short, proc=i, nprocx=nproc, time=tend) for i in range(nproc)]
        solver_plot.load(filepaths)
        energy.append(solver_plot.integrate(solver_plot.energy()))

        for (fig, axs), plot_fun, label in zip(fig_list, pfunc_list, labels):
            ax = axs[i // 2][i % 2]
            ax.tick_params(labelsize=8)

            if label == 'ice':
                # levels = np.linspace(0.0, 7e-3, 1000)
                levels = 1000
                cmap = cmap = cmocean.cm.ice
            elif label == 'entropy':
                # levels = np.linspace(-30, 70, 1000)
                levels = 1000
                cmap = cmap = cmocean.cm.thermal
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
            cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt))
            cbar.ax.tick_params(labelsize=8)

            if (i // 2) == 1:
                ax.set_xlabel('x (m)', fontsize='xx-small')
            if (i % 2) == 0:
                ax.set_ylabel('z (m)', fontsize='xx-small')
            # fig.tight_layout(w_pad=1.0, h_pad=1.0)
            fig.tight_layout()

    for (fig, ax), label in zip(fig_list, labels):
        plot_name = f'{label}_{exp_name_short}'
        fp = solver_plot.get_filepath(plot_dir, plot_name, ext='png')
        fig.savefig(fp, bbox_inches="tight")
