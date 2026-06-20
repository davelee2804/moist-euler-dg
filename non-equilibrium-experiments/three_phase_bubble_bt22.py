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
non_equilibrium_thermo = True

exp_name_short = 'bt22-prev_47'
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

vl_power_list = []
vi_power_list = []
li_power_list = []
v_water_list = []
l_water_list = []
i_water_list = []
entropy_list = []

def forcing_function(solver, state, dstatedt):
    u, w, h, s, qv, ql, qi = solver.get_vars(state)
    dudt, dwdt, dhdt, dsdt, dqvdt, dqldt, dqidt = solver.get_vars(dstatedt)

    #s_d = eta_k_to_eta_d(h, s, qv, ql, qi)

    # add heating terms in dsdt
    #T, mu_v, mu_l, mu_i = chem_pots(h, s_d, qv, ql, qi)
    _, T, _, _, mu_v, mu_l, mu_i = solver.get_thermodynamic_quantities(h, s, qv, ql, qi)

    if non_equilibrium_thermo:
        # parse therodynamic state to pytorch array
        scale = 1.0e+12
        hdt = 0.5 * solver.get_dt()
        nn_in = torch.from_numpy(np.array([qv.flatten(), \
                                           ql.flatten(), \
                                           qi.flatten(), \
                                           #s_d.flatten(), \
                                           s.flatten(), \
                                           h.flatten()]).transpose())
        # evaluate the nn and parse back as numpy array
        mp_incs = model(nn_in).detach().numpy().reshape([T.shape[0],T.shape[1],T.shape[2],T.shape[3],3])
        # solution increments from transport for checking monotonicity
        epsilon = 1.0e-14
        # vapor to liquid exchange
        u_dqv = qv + hdt * dqvdt
        u_dql = ql + hdt * dqldt
        u_dqi = qi + hdt * dqidt
        u_ds  = s  + hdt * dsdt
        inc    = mp_incs[:,:,:,:,0] * h * (mu_v - mu_l) / scale
        inc_s  = inc * (mu_v - mu_l) / T #/ scale
        use    = np.logical_and(u_dqv + hdt * inc > epsilon, u_dql - hdt * inc > epsilon)
        #use    = np.logical_and(use, u_ds - hdt * inc_s > epsilon)
        dqvdt += use * inc
        dqldt -= use * inc
        dsdt  -= use * inc_s
        vl_power_list.append(solver.integrate(use*h*inc_s*T))
        # vapor to ice exchange
        u_dqv = qv + hdt * dqvdt
        u_dql = ql + hdt * dqldt
        u_dqi = qi + hdt * dqidt
        u_ds  = s  + hdt * dsdt
        inc    = mp_incs[:,:,:,:,1] * h * (mu_v - mu_i) / scale
        inc_s  = inc * (mu_v - mu_i) / T #/ scale
        use    = np.logical_and(u_dqv + hdt * inc > epsilon, u_dqi - hdt * inc > epsilon)
        #use    = np.logical_and(use, u_ds - hdt * inc_s > epsilon)
        dqvdt += use * inc
        dqidt -= use * inc
        dsdt  -= use * inc_s
        vi_power_list.append(solver.integrate(use*h*inc_s*T))
        # liquid to ice exchange
        u_dqv = qv + hdt * dqvdt
        u_dql = ql + hdt * dqldt
        u_dqi = qi + hdt * dqidt
        u_ds  = s  + hdt * dsdt
        inc    = mp_incs[:,:,:,:,2] * h * (mu_l - mu_i) / scale
        inc_s  = inc * (mu_l - mu_i) / T #/ scale
        use    = np.logical_and(u_dql + hdt * inc > epsilon, u_dqi - hdt * inc > epsilon)
        #use    = np.logical_and(use, u_ds - hdt * inc_s > epsilon)
        dqldt += use * inc
        dqidt -= use * inc
        dsdt  -= use * inc_s
        li_power_list.append(solver.integrate(use*h*inc_s*T))

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

def get_vertical_data(a):
    nxe = a.shape[0]
    nze = a.shape[1]
    nxq = a.shape[2]
    nzq = a.shape[3]
    az = np.zeros(nze*nzq)
    for ii in np.arange(nze):
        az[ii*nzq:(ii+1)*nzq] = a[nxe//2,ii,nxq//2,:]
    return az

def plot_profile(z,f,field_name):
    plt.figure()
    plt.plot(f,z/1000.0)
    plt.title('initial profile: ' + field_name)
    plt.ylabel('z (km)')
    plt.savefig('initial_profile_'+field_name+'.png')

def initial_condition(xs, ys, solver, pert):
    gravity = 9.81

    u = 0.0 * ys
    v = 0.0 * ys

    theta_t = 3.3e-3 * (ys - 1000.0) + 300.0
    top = ys > 1000.0
    bot = ys <= 1000.0
    theta = top * theta_t + bot * 300.0

    rad = np.sqrt((xs / 2000.0) ** 2 + (ys / 500.0) ** 2)
    mask = rad < 1.0
    buoy_prime = mask * (pert * gravity / 300.0) * np.cos(0.5 * np.pi * rad) * np.cos(0.5 * np.pi * rad)
    theta += (300.0 / gravity) * buoy_prime

    # compute the exner pressure via d\Pi = -(g / c_p / \theta) dz
    ex = np.ones(ys.shape)
    nze = ys.shape[1]
    z = get_vertical_data(ys)
    for ze in np.arange(ys.shape[1]):
        for zq in np.arange(ys.shape[3]):
            if ze == 0 and zq == 0:
                continue
            elif zq == 0:
                dz = 0.0
            else:
                dz = ys[0,ze,0,zq] - ys[0,ze,0,zq-1]

            for xe in np.arange(ys.shape[0]):
                for xq in np.arange(ys.shape[2]):
                    if zq == 0:
                        ex[xe,ze,xq,zq] = ex[xe,ze-1,xq,-1]
                    else:
                        ex[xe,ze,xq,zq] = ex[xe,ze,xq,zq-1] - dz * gravity / solver.cpd / theta[xe,ze,xq,zq]

    exz = get_vertical_data(ex)
    plot_profile(z,exz,'exner')
    thz = get_vertical_data(theta)
    plot_profile(z,thz,'theta')

    # compute the density
    p = 1_00_000.0 * ex ** (solver.cpd / solver.Rd)
    density = p / (solver.Rd * ex * theta)

    qw = solver.rh_to_qw(0.7, p, density)
    qw = bot * 0.0125 + top * qw
    qd = 1 - qw

    R = solver.Rd * qd + solver.Rv * qw
    T = p / (R * density)

    #assert (qw <= solver.saturation_fraction(T, density)).all()

    s = qd * solver.entropy_air(T, qd, density)
    s += qw * solver.entropy_vapour(T, qw, density)

    #qv, ql, qi = solver.solve_fractions_from_entropy(density, qw, s)
    qv = 1.0 * qw
    ql = 0.0 * qw 
    qi = 0.0 * qw 
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

def plot_initial_profiles(solver):
    z = get_vertical_data(solver.zs)
    h = get_vertical_data(solver.h)
    s = get_vertical_data(solver.s)
    qv = get_vertical_data(solver.qv)
    ql = get_vertical_data(solver.ql)
    qi = get_vertical_data(solver.qi)
    plot_profile(z,h,'density')
    plot_profile(z,s,'entropy')
    plot_profile(z,qv,'vapor')
    plot_profile(z,ql,'liquid')
    plot_profile(z,qi,'ice')
    theta = pot_temp(h, s, qv, ql, qi)
    plot_profile(z,theta,'potential_temperature')

run_time = 2250
tends = np.array([0.0, 450.0, 900.0, 1400.0, 1800.0, 2250.0])

conservation_data_fp = os.path.join(data_dir, 'conservation_data.npy')
time_list = []
energy_list = []
entropy_var_list = []
water_var_list = []

if non_equilibrium_thermo:
    model_path = '/g/data/dp9/dl9118/lfric_ral_training/runs/nAdv_nEta_wBatch/'+exp_name_short[-7:]+'/model_min_loss.pt'
    model = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
    model.eval()

if run_model:
    solver = _NonEqEuler2D(
        xmap, zmap, poly_order, nx,
        g=g, cfl=cfl, a=a, nz=nz, upwind=upwind, nprocx=nproc,
        forcing=forcing_function
    )
    u, v, density, s, qw, qv, ql, qi = initial_condition(solver.xs, solver.zs, solver, pert=2.0)
    solver.set_initial_condition(u, v, density, s, qv, ql, qi)
    plot_initial_profiles(solver)
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
        conservation_data = np.zeros((11, len(time_list)))
        conservation_data[0, :] = np.array(time_list)
        conservation_data[1, :] = np.array(energy_list)
        conservation_data[2, :] = np.array(entropy_var_list)
        conservation_data[3, :] = np.array(water_var_list)
        conservation_data[4, :] = np.array(vl_power_list)[::8]
        conservation_data[5, :] = np.array(vi_power_list)[::8]
        conservation_data[6, :] = np.array(li_power_list)[::8]
        conservation_data[7, :] = np.array(v_water_list)[::8]
        conservation_data[8, :] = np.array(l_water_list)[::8]
        conservation_data[9, :] = np.array(i_water_list)[::8]
        conservation_data[10,:] = np.array(entropy_list)[::8]
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
    vl_power = conservation_data[4, :][mask]
    vi_power = conservation_data[5, :][mask]
    li_power = conservation_data[6, :][mask]
    v_water = conservation_data[7, :][mask]
    l_water = conservation_data[8, :][mask]
    i_water = conservation_data[9, :][mask]
    entropy = conservation_data[10,:][mask]
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
    plt.plot(time_list[1:], vl_power[1:], label='Vapor-liquid power')
    plt.plot(time_list[1:], vi_power[1:], label='Vapor-ice power')
    plt.plot(time_list[1:], li_power[1:], label='Liquid-ice power')
    plt.plot(time_list[1:], vl_power[1:] + vi_power[1:] + li_power[1:], label='Total')
    plt.grid()
    plt.legend()
    plt.ylabel('Watts (J/s)')
    plt.xlabel('Time (s)')
    plt.yscale('symlog', linthresh=1e-15)
    fp = os.path.join(plot_dir, f'power_{exp_name_short}')
    plt.savefig(fp, bbox_inches="tight")

    plt.figure()
    plt.plot(time_list[1:], v_water[1:], label='Vapor')
    plt.plot(time_list[1:], l_water[1:], label='Liquid')
    plt.plot(time_list[1:], i_water[1:], label='Ice')
    plt.plot(time_list[1:], (entropy[1:]-entropy[0])/entropy[0], label='Entropy change (normalised)')
    plt.grid()
    plt.legend()
    plt.ylabel('Mass (kg)')
    plt.xlabel('Time (s)')
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
