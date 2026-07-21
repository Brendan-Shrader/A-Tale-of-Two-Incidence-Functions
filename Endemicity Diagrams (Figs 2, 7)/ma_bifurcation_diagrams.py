import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

FONT_SIZE = 16
LEGEND_FONT_SIZE = 14

plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# Mass-action SIR model equations
def SIR_model(t, x, pars):
    Sp, I, Ss = x
    Lambda, mu, beta, gamma, epsilon, alpha_i, alpha_s = pars.values()

    dSp_dt = Lambda - beta*Sp*I - mu*Sp
    dI_dt = beta*Sp*I + epsilon*beta*Ss*I - (gamma + mu + alpha_i)*I
    dSs_dt = gamma*I - epsilon*beta*Ss*I - (mu + alpha_s)*Ss

    return np.array([dSp_dt, dI_dt, dSs_dt])

# Returns R0
def get_R0(pars):
    Lambda, mu, beta, gamma, epsilon, alpha_i, alpha_s = pars.values()
    return (Lambda / mu) * (beta / (gamma + mu + alpha_i))

# Returns Rε given β and other parameters
def R_epsilon_curve_beta(beta, pars):
    Lambda, mu, _, gamma, _, alpha_i, alpha_s = pars.values()
    return (gamma + mu + alpha_i) * ((mu + alpha_s) / Lambda) * (1 / beta)

# Returns Rε given α_s and other parameters
def R_epsilon_curve_alpha_s(alpha_s, pars):
    Lambda, mu, beta, gamma, _, alpha_i, _ = pars.values()
    return (gamma + mu + alpha_i) * ((mu + alpha_s) / Lambda) * (1 / beta)

# Generate a matrix of I* values for a grid of parameter values
def generate_I_star_matrix(pars, par1_info, par2_info):
    par1_name, par1_values = par1_info
    par2_name, par2_values = par2_info

    I_star_matrix = np.zeros((len(par1_values), len(par2_values)))
    N_star_matrix = np.zeros((len(par1_values), len(par2_values)))

    for i, par1_value in enumerate(par1_values):
        for j, par2_value in enumerate(par2_values):
            pars[par1_name] = par1_value
            pars[par2_name] = par2_value
            odefun = lambda t, x: SIR_model(t, x, pars)

            tol = 1e-8

            if get_R0(pars) <= 1:
                I_star_matrix[i, j] = 0
                N_star_matrix[i, j] = pars['Lambda'] / pars['mu']
            else:
                objfun = lambda x: SIR_model(0, x, pars)
                success_flag = False
                while success_flag == False:
                    x0 = np.random.rand(3)
                    sol = root(objfun, x0)
                    if sol.success and np.all(sol.x >= 0) and sol.x[1] >= tol and np.sum(sol.x) <= pars['Lambda'] / pars['mu']:
                        success_flag = True
                I_star_matrix[i, j] = sol.x[1]  # Final value of I
                N_star_matrix[i, j] = sum(sol.x)

    return I_star_matrix, N_star_matrix

# Default parameter values
pars = {
    'Lambda': 0.0004,          # Birth rate
    'mu': 0.0004,              # Death rate
    'beta': 5,                 # Transmission rate
    'gamma': 1,                # Recovery rate
    'epsilon': 1,              # Modification factor for transmission
    'alpha_i': 0,           # Additional mortality due to infection
    'alpha_s': 0.0008           # Additional mortality due to susceptibility
}



'''
β-ε bifurcation diagram with α_s = 0.0008
'''

# Setup parameter values for the bifurcation diagram
R0_max = 10
beta_max = R0_max * pars['mu'] / pars['Lambda'] * (pars['gamma'] + pars['mu'] + pars['alpha_i'])
beta_values = np.linspace(0, beta_max, 25)
epsilon_values = np.linspace(0, 1, 25)

# Generate the bifurcation diagram data
I_star_matrix, N_star_matrix = generate_I_star_matrix(pars, ('beta', beta_values), ('epsilon', epsilon_values))

# Make the bifurcation diagram figure
fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=100)

X, Y = np.meshgrid(beta_values, epsilon_values, indexing='ij')
Z = I_star_matrix / N_star_matrix

beta_at_R0_equals_1 = pars['mu'] / pars['Lambda'] * (pars['gamma'] + pars['mu'] + pars['alpha_i'])

cf_levels = np.linspace(0, 1, 100)
cf = ax.contourf(X, Y, Z, levels=cf_levels, cmap='magma', zorder=-1)
ax.plot(beta_values[1:], R_epsilon_curve_beta(beta_values[1:], pars), color='red', label=r'$\mathcal{R}_\epsilon = 1$')
ax.plot([beta_at_R0_equals_1, beta_at_R0_equals_1], [0, 1], color='blue', label=r'$\mathcal{R}_0 = 1$')

ax.set_xlabel(r'$\mathcal{R}_0$', rotation=0, labelpad=5)
ax.set_ylabel(r'$\epsilon$', rotation=0, labelpad=15)
ax.set_xlim([0, beta_max])
ax.set_ylim([0, 1])

num_xticks = 6
xtick_labels = np.linspace(0, R0_max, num_xticks)
ax.set_xticks(np.linspace(0, beta_max, num_xticks))
ax.set_xticklabels(f"{xtick:.0f}" for xtick in xtick_labels)

ax.set_title(r'$\alpha_s = %.4f$' % pars['alpha_s'], fontsize=FONT_SIZE)

cbar = fig.colorbar(cf, label=r'$I^*$')
cbar_ticks = np.arange(0, 1.1, 0.2)
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([f'{tick:.1f}' for tick in cbar_ticks])
cbar.set_label(r'$\frac{I^*}{N^*}$', rotation=0, labelpad=20)

ax.legend(framealpha=1, fancybox=False, edgecolor='black', loc='lower right', fontsize=LEGEND_FONT_SIZE)

ax.set_rasterization_zorder(0)

filename = f'ma_beta-epsilon_bifurcation_diagram_alpha_s={pars["alpha_s"]:.4f}.pdf'
fig.savefig(filename, dpi=300, bbox_inches='tight')



'''
β-ε bifurcation diagram with α_s = 0
'''

# Setup parameter values for the bifurcation diagram
pars['alpha_s'] = 0
beta_values = np.linspace(0, beta_max, 25)
epsilon_values = np.linspace(0, 1, 25)

# Generate the bifurcation diagram data
I_star_matrix, N_star_matrix = generate_I_star_matrix(pars, ('beta', beta_values), ('epsilon', epsilon_values))

# Make the bifurcation diagram figure
fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=100)

X, Y = np.meshgrid(beta_values, epsilon_values, indexing='ij')
Z = I_star_matrix / N_star_matrix

beta_at_R0_equals_1 = pars['mu'] / pars['Lambda'] * (pars['gamma'] + pars['mu'] + pars['alpha_i'])

cf_levels = np.linspace(0, 1, 100)
cf = ax.contourf(X, Y, Z, levels=cf_levels, cmap='magma', zorder=-1)
ax.plot(beta_values[1:], R_epsilon_curve_beta(beta_values[1:], pars), color='red', label=r'$\mathcal{R}_\epsilon = 1$')
ax.plot([beta_at_R0_equals_1, beta_at_R0_equals_1], [0, 1], color='blue', label=r'$\mathcal{R}_0 = 1$')

ax.set_xlabel(r'$\mathcal{R}_0$', rotation=0, labelpad=5)
ax.set_ylabel(r'$\epsilon$', rotation=0, labelpad=15)
ax.set_xlim([0, beta_max])
ax.set_ylim([0, 1])

num_xticks = 6
xtick_labels = np.linspace(0, R0_max, num_xticks)
ax.set_xticks(np.linspace(0, beta_max, num_xticks))
ax.set_xticklabels(f"{xtick:.0f}" for xtick in xtick_labels)

ax.set_title(r'$\alpha_s = %.4f$' % pars['alpha_s'], fontsize=FONT_SIZE)

cbar = fig.colorbar(cf, label=r'$I^*$')
cbar_ticks = np.arange(0, 1.1, 0.2)
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([f'{tick:.1f}' for tick in cbar_ticks])
cbar.set_label(r'$\frac{I^*}{N^*}$', rotation=0, labelpad=20)

ax.legend(framealpha=1, fancybox=False, edgecolor='black', loc='upper right', fontsize=LEGEND_FONT_SIZE)

ax.set_rasterization_zorder(0)

filename = f'ma_beta-epsilon_bifurcation_diagram_alpha_s={pars["alpha_s"]:.4f}.pdf'
fig.savefig(filename, dpi=300, bbox_inches='tight')




'''
β-ε bifurcation diagram with α_s = 0.0002 without R_epsilon curve
'''

# Setup parameter values for the bifurcation diagram
pars['alpha_s'] = 0.0002
beta_values = np.linspace(0, beta_max, 25)
epsilon_values = np.linspace(0, 1, 25)

# Generate the bifurcation diagram data
I_star_matrix, N_star_matrix = generate_I_star_matrix(pars, ('beta', beta_values), ('epsilon', epsilon_values))

# Make the bifurcation diagram figure
fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=100)

X, Y = np.meshgrid(beta_values, epsilon_values, indexing='ij')
Z = I_star_matrix / N_star_matrix

beta_at_R0_equals_1 = pars['mu'] / pars['Lambda'] * (pars['gamma'] + pars['mu'] + pars['alpha_i'])

cf_levels = np.linspace(0, 1, 100)
cf = ax.contourf(X, Y, Z, levels=cf_levels, cmap='magma', zorder=-1)
#ax.plot(beta_values[1:], R_epsilon_curve_beta(beta_values[1:], pars), color='red', label=r'$\mathcal{R}_\epsilon = 1$')
ax.plot([beta_at_R0_equals_1, beta_at_R0_equals_1], [0, 1], color='blue', label=r'$\mathcal{R}_0 = 1$')

ax.set_xlabel(r'$\mathcal{R}_0$', rotation=0, labelpad=5)
ax.set_ylabel(r'$\epsilon$', rotation=0, labelpad=15)
ax.set_xlim([0, beta_max])
ax.set_ylim([0, 1])

num_xticks = 6
xtick_labels = np.linspace(0, R0_max, num_xticks)
ax.set_xticks(np.linspace(0, beta_max, num_xticks))
ax.set_xticklabels(f"{xtick:.0f}" for xtick in xtick_labels)

ax.set_title("Mass-Action Incidence", fontsize=FONT_SIZE)

cbar = fig.colorbar(cf, label=r'$I^*$')
cbar_ticks = np.arange(0, 1.1, 0.2)
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([f'{tick:.1f}' for tick in cbar_ticks])
cbar.set_label(r'$\frac{I^*}{N^*}$', rotation=0, labelpad=20)

ax.legend(framealpha=1, fancybox=False, edgecolor='black', loc='upper right', fontsize=LEGEND_FONT_SIZE)

ax.set_rasterization_zorder(0)

filename = f'ma_beta-epsilon_bifurcation_diagram_alpha_s={pars["alpha_s"]:.4f}_no_R_epsilon.pdf'
fig.savefig(filename, dpi=300, bbox_inches='tight')