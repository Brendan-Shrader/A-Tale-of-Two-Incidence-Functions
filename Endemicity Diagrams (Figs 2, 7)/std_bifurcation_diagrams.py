import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root

FONT_SIZE = 16
LEGEND_FONT_SIZE = 14

plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

def si_model(t, x, pars):
    s, i = x
    b, beta, gamma, epsilon, alpha_i, alpha_s = pars.values()

    ds_dt = b - b*s - beta*s*i + alpha_i*s*i + alpha_s*s*(1-s-i)
    di_dt = beta*s*i + epsilon*beta*i*(1-s-i) - (gamma + b + alpha_i)*i + alpha_i*i**2 + alpha_s*i*(1-s-i)

    return np.array([ds_dt, di_dt])

def get_R0(pars):
    b, beta, gamma, epsilon, alpha_i, alpha_s = pars.values()
    return beta / (gamma + b + alpha_i)

def generate_I_star_matrix(pars, par1_info, par2_info):
    par1_name, par1_values = par1_info
    par2_name, par2_values = par2_info

    i_star_matrix = np.zeros((len(par1_values), len(par2_values)))

    for i, par1_value in enumerate(par1_values):
        for j, par2_value in enumerate(par2_values):
            pars[par1_name] = par1_value
            pars[par2_name] = par2_value
            odefun = lambda t, x: si_model(t, x, pars)

            tol = 1e-8

            if get_R0(pars) <= 1:
                i_star_matrix[i, j] = 0
            else:
                objfun = lambda x: si_model(0, x, pars)
                success_flag = False
                while success_flag == False:
                    x0 = np.random.rand(2)
                    sol = root(objfun, x0)
                    if sol.success and np.all(sol.x >= 0) and sol.x[1] >= tol and np.sum(sol.x) <= 1:
                        success_flag = True
                i_star_matrix[i, j] = sol.x[1]  # Final value of I

    return i_star_matrix

pars = {
    'b': 0.0004,
    'beta': 5,                 # Transmission rate
    'gamma': 1,                # Recovery rate
    'epsilon': 1,              # Modification factor for transmission
    'alpha_i': 0,           # Additional mortality due to infection
    'alpha_s': 0.0002           # Additional mortality due to susceptibility
}



'''
β-ε bifurcation diagram with α_s = 0.0008
'''
R0_max = 10
beta_max = R0_max * (pars['gamma'] + pars['b'] + pars['alpha_i'])
beta_values = np.linspace(0, beta_max, 25)
epsilon_values = np.linspace(0, 1, 25)
I_star_matrix = generate_I_star_matrix(pars, ('beta', beta_values), ('epsilon', epsilon_values))
fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=100)

X, Y = np.meshgrid(beta_values, epsilon_values, indexing='ij')
Z = I_star_matrix

beta_at_R0_equals_1 = pars['gamma'] + pars['b'] + pars['alpha_i']

#cf_levels = np.linspace(0, 1, 100)
cf_levels = np.linspace(0, 1, 100)
cf = ax.contourf(X, Y, Z, levels=cf_levels, cmap='magma')
ax.plot([beta_at_R0_equals_1, beta_at_R0_equals_1], [0, 1], color='blue', label=r'$R_0 = 1$')

ax.set_xlabel(r'$\mathcal{R}_0$', rotation=0, labelpad=5)
ax.set_ylabel(r'$\epsilon$', rotation=0, labelpad=15)
ax.set_xlim([0, beta_max])
ax.set_ylim([0, 1])

num_xticks = 6
xtick_labels = np.linspace(0, R0_max, num_xticks)
ax.set_xticks(np.linspace(0, beta_max, num_xticks))
ax.set_xticklabels(f"{xtick:.0f}" for xtick in xtick_labels)

ax.set_title("Standard Incidence", fontsize=FONT_SIZE)

cbar = fig.colorbar(cf, label=r'$I^*$')
cbar_ticks = np.arange(0, 1.1, 0.2)
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([f'{tick:.1f}' for tick in cbar_ticks])
cbar.set_label(r'$i^*$', rotation=0, labelpad=20)

ax.legend(framealpha=1, fancybox=False, edgecolor='black', loc='upper right', fontsize=LEGEND_FONT_SIZE)

filename = f'std_beta-epsilon_bifurcation_diagram_alpha_s={pars["alpha_s"]:.4f}_no_R_epsilon.png'
fig.savefig(filename, dpi=300, bbox_inches='tight')