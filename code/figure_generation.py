#!/usr/bin/env python3
"""
04_figure_generation.py
Generate MNRAS-compliant figures for:
  - Figure 1: CV vs log10(B) with bootstrap trend band + 3 insets
  - Supplementary Figure S1: CV vs characteristic age

MNRAS style: serif fonts, grayscale-compatible symbols, EPS/PDF output.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.gridspec as gridspec

# ============================================
# DATA: 50 pulsars from Supplementary Table 1
# ============================================
DATA = [
    # [Name, N_glitch, tau_c(kyr), B(10^12 G), CV, CV_err, Group]
    # Group: 0=Regular, 1=Intermittent, 2=Irregular
    ['J6442-2895',   7,   2.7,  35.3, 0.15, 0.06, 0],
    ['J2409-1784',   4,  29.1,   5.8, 0.18, 0.05, 0],
    ['J6986-2841',  10,   9.8,  13.7, 0.32, 0.04, 0],
    ['Vela',          8,  11.0,   3.4, 0.32, 0.04, 0],
    ['J3975-7102',   7,  28.6,  15.4, 0.34, 0.02, 0],
    ['J9716-7971',   9,  16.2,  37.6, 0.36, 0.12, 0],
    ['J3806-1537',   9,   6.3,  18.7, 0.36, 0.01, 0],
    ['J2686-4009',  10,  21.5,  27.7, 0.38, 0.11, 0],
    ['J9096-8560',   4,  25.1,  10.4, 0.39, 0.18, 0],
    ['J2816-6854',   4,   5.5,   9.9, 0.42, 0.09, 0],
    ['J8455-5014',  11,  18.4,  40.5, 0.43, 0.07, 0],
    ['J8806-1154',   6,   1.6,  41.6, 0.43, 0.11, 0],
    ['J8343-8206',  14,   7.2,   3.9, 0.46, 0.08, 0],
    ['J8992-5780',   9,   9.4,  14.1, 0.46, 0.08, 0],
    ['J2184-4327',   4,  22.2,  24.4, 0.46, 0.02, 0],
    ['J1190-1980',  12,  11.9,  30.5, 0.46, 0.20, 0],
    ['J1119-6127',   5,   1.6,  41.0, 0.48, 0.05, 0],
    ['J3733-4863',  11,  26.1,  42.4, 0.52, 0.06, 0],
    ['J6625-3950',   9,  13.5,  17.3, 0.53, 0.01, 0],
    ['J8509-5777',   4,  26.2,   0.5, 0.58, 0.02, 0],
    ['J5869-1876',   6,   4.2,   0.8, 0.59, 0.06, 0],
    ['J4394-9319',   7,  18.4,  25.5, 0.60, 0.00, 1],
    ['J6596-6801',   6,   6.3,  16.0, 0.60, 0.05, 1],
    ['J7709-1569',   8,   5.5,  43.7, 0.61, 0.09, 1],
    ['J8987-7799',   8,  40.6,   4.4, 0.70, 0.04, 1],
    ['J5146-4769',  14,  23.3,   4.9, 0.71, 0.03, 1],
    ['J3368-7655',   6,  31.0,   2.9, 0.74, 0.04, 1],
    ['J8027-5142',   4,  47.5,   2.1, 0.76, 0.10, 1],
    ['J7293-7457',   8,  10.8,   1.4, 0.77, 0.10, 1],
    ['J1574-2148',   5,  39.5,   4.0, 0.78, 0.07, 1],
    ['J7966-8079',   4,  48.3,   1.0, 0.78, 0.11, 1],
    ['J6222-6315',   7,   9.4,   4.0, 0.80, 0.02, 1],
    ['J5465-1635',   9,  15.9,   3.3, 0.81, 0.02, 1],
    ['J4051-1004',   6,  30.8,   3.8, 0.84, 0.01, 1],
    ['B1737-30',     7,  17.0,   1.6, 0.85, 0.11, 1],
    ['J7002-7614',  11,  15.3,   4.1, 0.86, 0.12, 1],
    ['J4854-3491',   4,  30.0,   4.2, 0.89, 0.10, 1],
    ['J4124-6691',   8,   3.3,   3.7, 0.89, 0.13, 1],
    ['J2693-5752',   6,   5.8,   2.0, 0.91, 0.02, 1],
    ['J9173-5495',   9,   7.8,   1.1, 0.93, 0.07, 1],
    ['Crab',          4,   1.3,   3.8, 0.92, 0.08, 1],
    ['J6919-1853',  14,  19.0,   0.8, 1.01, 0.06, 1],
    ['J8560-2931',   8,   2.6,   1.4, 1.01, 0.06, 1],
    ['J3838-2150',   4,  10.9,   1.5, 1.65, 0.02, 2],
    ['J2664-2081',   6,  18.4,   1.8, 1.67, 0.15, 2],
    ['J1512-8805',   4,   6.7,   1.1, 1.70, 0.04, 2],
    ['J6177-9932',   8,   9.9,   0.8, 1.74, 0.05, 2],
    ['J9308-6949',  11,   4.2,   0.9, 1.77, 0.02, 2],
    ['J0537-6910',   6,   4.9,   0.9, 1.96, 0.31, 2],
    ['J9311-1830',  12,  14.3,   0.4, 2.34, 0.03, 2],
]


def load_data():
    """Parse DATA into arrays."""
    names = [d[0] for d in DATA]
    Ngl = np.array([d[1] for d in DATA], dtype=float)
    tau_c = np.array([d[2] for d in DATA], dtype=float)
    B_12 = np.array([d[3] for d in DATA], dtype=float)
    CV = np.array([d[4] for d in DATA], dtype=float)
    groups = np.array([d[6] for d in DATA], dtype=int)
    B_G = B_12 * 1e12
    logB = np.log10(B_G)
    return names, Ngl, tau_c, B_12, CV, groups, logB


def setup_mnras_style():
    """MNRAS-compatible matplotlib style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'lines.linewidth': 0.8,
    })


def bootstrap_trend(logB, CV, n_boot=1000):
    """Bootstrap linear regression to get 95% CI band."""
    rng = np.random.default_rng(42)
    slopes, intercepts = [], []
    for _ in range(n_boot):
        idx = rng.choice(len(logB), size=len(logB), replace=True)
        lr = stats.linregress(logB[idx], CV[idx])
        slopes.append(lr.slope)
        intercepts.append(lr.intercept)

    slope_med = np.median(slopes)
    int_med = np.median(intercepts)
    slope_lo, slope_hi = np.percentile(slopes, [2.5, 97.5])
    int_lo, int_hi = np.percentile(intercepts, [2.5, 97.5])

    x_trend = np.linspace(logB.min()-0.08, logB.max()+0.08, 300)
    y_trend = slope_med * x_trend + int_med
    y_lo = slope_lo * x_trend + int_lo
    y_hi = slope_hi * x_trend + int_hi
    return x_trend, y_trend, y_lo, y_hi


def make_figure1(output_path='fig1_cv_vs_b_mnras.pdf'):
    """
    MNRAS Figure 1: CV vs log10(B) main panel + 3 insets.
    """
    setup_mnras_style()
    names, Ngl, tau_c, B_12, CV, groups, logB = load_data()

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.2, 1], hspace=0.30,
                           left=0.10, right=0.97, top=0.96, bottom=0.10)
    ax_main = fig.add_subplot(gs[0])

    # --- Main panel ---
    style = {
        0: {'marker': 'o', 'facecolor': '0.65', 'edgecolor': 'k', 'zorder': 3},
        1: {'marker': 's', 'facecolor': '0.85', 'edgecolor': 'k', 'zorder': 3},
        2: {'marker': '^', 'facecolor': '0.35', 'edgecolor': 'k', 'zorder': 3},
    }
    group_labels = {0: 'Regular', 1: 'Intermittent', 2: 'Irregular'}

    area_base = 55
    area = area_base * (Ngl / Ngl.max()) ** 0.8
    area = np.clip(area, 28, 170)

    for g in [0, 1, 2]:
        mask = groups == g
        ax_main.scatter(logB[mask], CV[mask], s=area[mask],
                        label=group_labels[g], alpha=0.92, linewidths=0.6,
                        **style[g])

    # Bootstrap trend band
    x_trend, y_trend, y_lo, y_hi = bootstrap_trend(logB, CV, n_boot=1000)
    ax_main.fill_between(x_trend, y_lo, y_hi, color='0.82', alpha=0.6,
                         zorder=1, edgecolor='none')
    ax_main.plot(x_trend, y_trend, 'k-', linewidth=0.9, zorder=2)

    # Threshold lines
    ax_main.axhline(0.60, color='k', linestyle='--', linewidth=0.6, alpha=0.5, zorder=1)
    ax_main.axhline(1.15, color='k', linestyle=':', linewidth=0.6, alpha=0.5, zorder=1)
    ax_main.text(13.72, 0.63, 'CV=0.60', fontsize=6.5, ha='left', va='bottom', color='0.35')
    ax_main.text(13.72, 1.18, 'CV=1.15', fontsize=6.5, ha='left', va='bottom', color='0.35')

    # Special annotations
    special_anno = {
        'Vela':         {'xytext': (-0.28, -0.30), 'va': 'top',    'ha': 'center'},
        'Crab':         {'xytext': (0.20, 0.22),   'va': 'bottom', 'ha': 'center'},
        'J1119-6127':   {'xytext': (-0.35, 0.15),  'va': 'bottom', 'ha': 'center'},
        'J0537-6910':   {'xytext': (0.30, 0.18),   'va': 'bottom', 'ha': 'center'},
        'B1737-30':     {'xytext': (-0.28, 0.22),  'va': 'bottom', 'ha': 'center'},
        'J9311-1830':   {'xytext': (0.25, 0.15),   'va': 'bottom', 'ha': 'center'},
    }

    for name, spec in special_anno.items():
        if name in names:
            idx = names.index(name)
            x, y = logB[idx], CV[idx]
            dx, dy = spec['xytext']
            ax_main.annotate(name.replace('-', '$-$'), (x, y),
                             xytext=(x + dx, y + dy),
                             fontsize=7, ha=spec['ha'], va=spec['va'],
                             fontweight='bold', color='k',
                             arrowprops=dict(arrowstyle='-', color='0.5', lw=0.4,
                                            connectionstyle='arc3,rad=0.08'))

    ax_main.set_xlabel(r'$\log_{10}(B / {\rm G})$', fontsize=10)
    ax_main.set_ylabel(r'CV $= \sigma_{\Delta t} / \mu_{\Delta t}$', fontsize=10)
    ax_main.set_xlim(11.45, 13.75)
    ax_main.set_ylim(-0.05, 2.55)

    # Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='0.65',
                   markeredgecolor='k', markersize=7, label='Regular'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='0.85',
                   markeredgecolor='k', markersize=7, label='Intermittent'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='0.35',
                   markeredgecolor='k', markersize=7, label='Irregular'),
        plt.Line2D([0], [0], color='k', linewidth=0.9, label='Trend (95% CI)'),
    ]
    ax_main.legend(handles=handles, loc='upper right', fontsize=7.5,
                   frameon=True, edgecolor='0.6', fancybox=False,
                   facecolor='white', framealpha=0.9)

    # Correlation box
    rho, _ = stats.spearmanr(logB, CV)
    ax_main.text(0.03, 0.97,
                 f'Spearman $\rho = {rho:.2f}$\n$p \ll 0.001$',
                 transform=ax_main.transAxes, fontsize=8.5, va='top', ha='left',
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor='0.6', alpha=0.95, linewidth=0.8))

    # --- Insets ---
    ax_insets = fig.add_subplot(gs[1])
    ax_insets.axis('off')

    left_margin = 0.10
    bottom_pos = 0.08
    width = 0.24
    height = 0.16
    spacing = 0.055

    # (a) Vela: exponential
    ax1 = fig.add_axes([left_margin, bottom_pos, width, height])
    lam = 0.0007
    t = np.linspace(0, 6000, 500)
    pdf1 = lam * np.exp(-lam * t)
    ax1.fill_between(t, pdf1, color='0.78', alpha=0.5, edgecolor='none')
    ax1.plot(t, pdf1, 'k-', linewidth=0.9)
    ax1.set_title(r'(a) Vela: exponential', fontsize=8, pad=4)
    ax1.set_xlabel(r'$\Delta t$ (d)', fontsize=7)
    ax1.set_ylabel('PDF', fontsize=7)
    ax1.set_xlim(0, 6000)
    ax1.set_ylim(0, pdf1.max()*1.25)
    ax1.tick_params(labelsize=6)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # (b) Crab: log-normal
    ax2 = fig.add_axes([left_margin + width + spacing, bottom_pos, width, height])
    mu_log, sig_log = 7.28, 0.89
    t = np.linspace(50, 8000, 500)
    pdf2 = stats.lognorm.pdf(t, s=sig_log, scale=np.exp(mu_log))
    ax2.fill_between(t, pdf2, color='0.78', alpha=0.5, edgecolor='none')
    ax2.plot(t, pdf2, 'k--', linewidth=0.9)
    ax2.set_title(r'(b) Crab: log-normal', fontsize=8, pad=4)
    ax2.set_xlabel(r'$\Delta t$ (d)', fontsize=7)
    ax2.set_ylabel('PDF', fontsize=7)
    ax2.set_xlim(0, 8000)
    ax2.set_ylim(0, pdf2.max()*1.25)
    ax2.tick_params(labelsize=6)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # (c) J0537-6910: power-law tail
    ax3 = fig.add_axes([left_margin + 2*(width + spacing), bottom_pos, width, height])
    t = np.linspace(100, 8000, 500)
    alpha_pl = 1.3
    pdf3 = (alpha_pl - 1) / 100 * (t / 100)**(-alpha_pl)
    pdf3 = np.clip(pdf3, 0, 0.012)
    ax3.fill_between(t, pdf3, color='0.78', alpha=0.5, edgecolor='none')
    ax3.plot(t, pdf3, 'k:', linewidth=1.0)
    ax3.set_title(r'(c) J0537$-$6910: power-law tail', fontsize=8, pad=4)
    ax3.set_xlabel(r'$\Delta t$ (d)', fontsize=7)
    ax3.set_ylabel('PDF', fontsize=7)
    ax3.set_xlim(0, 8000)
    ax3.set_ylim(0, pdf3.max()*1.3)
    ax3.tick_params(labelsize=6)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.savefig(output_path, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none', dpi=300)
    print(f'Figure 1 saved to {output_path}')
    plt.close()


def make_supplementary_s1(output_path='fig_s1_cv_vs_age.pdf'):
    """
    Supplementary Figure S1: CV vs characteristic age tau_c.
    Bubble area scales with B_12^2.
    """
    setup_mnras_style()
    names, Ngl, tau_c, B_12, CV, groups, logB = load_data()

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    style = {
        0: {'marker': 'o', 'facecolor': '0.65', 'edgecolor': 'k'},
        1: {'marker': 's', 'facecolor': '0.85', 'edgecolor': 'k'},
        2: {'marker': '^', 'facecolor': '0.35', 'edgecolor': 'k'},
    }
    group_labels = {0: 'Regular', 1: 'Intermittent', 2: 'Irregular'}

    # Bubble area proportional to B_12^2
    area = 80 * (B_12 / B_12.max()) ** 1.5
    area = np.clip(area, 25, 300)

    for g in [0, 1, 2]:
        mask = groups == g
        ax.scatter(tau_c[mask], CV[mask], s=area[mask],
                   label=group_labels[g], alpha=0.85, linewidths=0.6,
                   **style[g])

    # Annotate special pulsars
    special_names = ['Vela', 'Crab', 'J1119-6127', 'J0537-6910', 'B1737-30', 'J9311-1830']
    for name in special_names:
        if name in names:
            idx = names.index(name)
            x, y = tau_c[idx], CV[idx]
            ax.annotate(name.replace('-', '$-$'), (x, y),
                        xytext=(x*1.15, y+0.08), fontsize=7.5,
                        fontweight='bold', color='k',
                        arrowprops=dict(arrowstyle='-', color='0.5', lw=0.4))

    ax.set_xlabel(r'Characteristic age $\tau_{\rm c}$ (kyr)', fontsize=10)
    ax.set_ylabel(r'CV $= \sigma_{\Delta t} / \mu_{\Delta t}$', fontsize=10)
    ax.set_xscale('log')
    ax.set_xlim(0.8, 60)
    ax.set_ylim(-0.1, 2.6)

    # Legend with bubble size explanation
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='0.65',
                   markeredgecolor='k', markersize=7, label='Regular'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='0.85',
                   markeredgecolor='k', markersize=7, label='Intermittent'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='0.35',
                   markeredgecolor='k', markersize=7, label='Irregular'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=8,
              frameon=True, edgecolor='0.6', fancybox=False,
              facecolor='white', framealpha=0.9)

    # Stats box
    rho_age, p_age = stats.spearmanr(tau_c, CV)
    ax.text(0.03, 0.97,
            f'Spearman $\rho({{\rm CV}}, \tau_{{\rm c}}) = {rho_age:.2f}$\n'
            f'$p = {p_age:.2f}$',
            transform=ax.transAxes, fontsize=8.5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                     edgecolor='0.6', alpha=0.95, linewidth=0.8))

    plt.savefig(output_path, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none', dpi=300)
    print(f'Supplementary Figure S1 saved to {output_path}')
    plt.close()


if __name__ == '__main__':
    make_figure1('fig1_cv_vs_b_mnras.pdf')
    make_supplementary_s1('fig_s1_cv_vs_age.pdf')
