# -*- coding: utf-8 -*-
#
#       Copyright 2022
#       Maximiliano Isi <max.isi@ligo.org>
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.

from pylab import *
import seaborn as sns
import pandas as pd
import ringdown as rd

# rcParams

# make plots fit the LaTex column size but rescale them for ease of display
scale_factor = 2

# Get columnsize from LaTeX using \showthe\columnwidth
fig_width_pt = scale_factor*246.0
# Convert pts to inches
inches_per_pt = 1.0/72.27               
# Golden ratio
fig_ratio = (np.sqrt(5)-1.0)/2.0
fig_width = fig_width_pt*inches_per_pt
fig_height =fig_width*fig_ratio

figsize_column = (fig_width, fig_height)
figsize_square = (fig_width, fig_width)

fig_width_page = scale_factor*inches_per_pt*508.87
figsize_page = (fig_width_page, fig_height)

rcParams = {'figure.figsize': figsize_column}

# LaTex text font sizse in points (rescaled as above)
fs = scale_factor*9
fs_label = 0.8*fs


def plot_pair_violin(dfs, keys=None, sr=2048, figsize=(10,3)):
    figure(figsize=figsize)

    if keys is None:
        keys = dfs.keys()

    df = pd.DataFrame()
    for k in keys:
        rdf = dfs[k][dfs[k]['srate']==sr].copy()
        rdf['run'] = k
        df = df.append(rdf)

    g = sns.violinplot(x='$t_0/M$', y="$A_1$", hue="run",
                       data=df, palette="Set2", split=True,
                       inner="quartile")
    axhline(0, ls='--', c='k', alpha=0.5)
    new_xlabels = ['{:.2f}'.format(float(i.get_text())) for i in g.get_xticklabels()]
    xticks(g.get_xticks(), new_xlabels);
    title('analysis srate {} Hz'.format(sr));
    legend(loc='upper left');
    return g

def plot_sigmas(dfs, keys=None, sr=2048, figsize=(10,3), ax=None, c=None,
                label=None, m_ref=69, chi_ref=0.69):
    tM = m_ref*rd.qnms.T_MSUN
    _, tau = rd.qnms.get_ftau(m_ref, chi_ref, 1)
    tauM = tau / tM

    if keys is None:
        keys = dfs.keys()

    if ax is None:
        fig, ax = subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    for i, key in enumerate(keys):
        df = dfs[key].get(sr, pd.DataFrame())
        if not df.empty:
            l = ax.errorbar(df.index,
                     df['med'], yerr=(df['med'] - df['lo'], df['hi'] - df['med']), fmt='.',
                     capsize=4, alpha=0.7, lw=2, capthick=2, label=label or key, color=c)
            cc = l.get_children()[0].get_color()
            # plot expo trendline
            A0 = df['med'][min(abs(df.index[df.index>=0]))]
            t = df.index.values
            ax.plot(t, A0*exp(-t/tauM), c=cc, alpha=0.5, ls='--')
            ax.plot(t[t>=0], A0*exp(-t[t>=0]/tauM), c=cc)
    ax.axhline(0, ls='--', c='k')
    ax.legend();
    ax.set_ylabel('$A_1$')
    ax.set_xlabel('$t_0/t_M$')
    ax.set_title('analysis sampling rate {}'.format(sr));
    return fig

def plot_amps(fit, truth=None, d=1, g=None, levels=[0.9, 0.5, 0.1], points=True,
              truth_kws=None, xlim=(0, 8E-21), ylim=(0, 14E-21), **kws):
    with sns.plotting_context('paper', font_scale=1.5):
        g = g or sns.JointGrid(x=[], y=[], xlim=xlim, ylim=ylim)
        g.x = 2*fit.posterior.A[:,::d,0].values.flatten()
        g.y = 2*fit.posterior.A[:,::d,1].values.flatten()
        # set style
        lws = kws.get('lws', linspace(1, 2, len(levels)))
        lkws = kws.get('lkws', dict(lw=lws[-1], ls=kws.get('ls', '-')))
        l, = g.ax_joint.plot([], [], label=kws.get('label', None),
                             c=kws.pop('c', kws.pop('color', None)),
                             **lkws)
        c = l.get_color()
        # plot
        if points:
            g.plot_joint(scatter, color=c, alpha=0.03, marker='.')
        g.plot_joint(rd.kdeplot_2d_clevels, colors=[c,], cmap=None, levels=levels,
                    linewidths=lws, linestyles=kws.get('ls', '-'), **kws)
        calpha = matplotlib.colors.to_rgba(c, kws.get('alpha', None))
        g.plot_marginals(sns.kdeplot, c=calpha, **lkws)
        
        if truth:
            tkws = dict(c=c, ls='--')
            tkws.update(truth_kws or {})
            
            g.ax_joint.axvline(truth['M'], **tkws)
            g.ax_joint.axhline(truth['chi'], **tkws)
            g.ax_joint.plot(truth['M'], truth['chi'], marker='+', markersize=10,
                            markeredgewidth=1.5, **tkws)
        g.set_axis_labels(r'$A_0$', r'$A_1$');
    return g

def plot_mchi(fit=None, x=None, y=None, truth=None, d=1, g=None, levels=[0.9, 0.5, 0.1], points=True,
              truth_kws=None, xlim=(50, 100), ylim=(0, 1), marginals=True, **kws):
    with sns.plotting_context('paper', font_scale=1.5):
        g = g or sns.JointGrid(x=[], y=[], xlim=xlim, ylim=ylim)
        g.x = fit.posterior.M[:,::d].values.flatten() if x is None else x
        g.y = fit.posterior.chi[:,::d].values.flatten() if y is None else y
        # set style
        lws = kws.get('lws', linspace(1, 2, len(levels)))
        lkws = kws.get('lkws', dict(lw=lws[-1], ls=kws.get('ls', '-')))
        l, = g.ax_joint.plot([], [], label=kws.get('label', None),
                             c=kws.pop('c', kws.pop('color', None)),
                             **lkws)
        c = l.get_color()
        # plot
        if points:
            g.plot_joint(scatter, color=c, alpha=0.03, marker='.')
        g.plot_joint(rd.kdeplot_2d_clevels, colors=[c,], cmap=None, levels=levels,
                    linewidths=lws, linestyles=kws.get('ls', '-'), **kws)
        calpha = matplotlib.colors.to_rgba(c, kws.get('alpha', None))
        if marginals:
            g.plot_marginals(sns.kdeplot, color=calpha, alpha=kws.get('alpha', None), **lkws)
        
        if truth:
            tkws = dict(c=c, ls='--')
            tkws.update(truth_kws or {})
            
            g.ax_joint.axvline(truth['M'], **tkws)
            g.ax_joint.axhline(truth['chi'], **tkws)
            g.ax_joint.plot(truth['M'], truth['chi'], marker='+', markersize=10,
                            markeredgewidth=1.5, **tkws)
        g.set_axis_labels(r'$M/M_\odot$', r'$\chi$');
    return g

def plot_dfdtau(fit, truth=None, d=1, g=None, levels=[0.9, 0.5, 0.1], points=True,
                  truth_kws=None, xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), npoints=1000, **kws):
    with sns.plotting_context('paper', font_scale=1.5):
        g = g or sns.JointGrid(x=[], y=[], xlim=xlim, ylim=ylim)
        n = fit.posterior.df.shape[0]*fit.posterior.df.shape[1]
        ixs = np.random.choice(n, min(n, npoints))
        g.x = fit.posterior.df[:,::d].values.flatten()[ixs]
        g.y = fit.posterior.dtau[:,::d].values.flatten()[ixs]
        # set style
        lws = kws.get('lws', linspace(1, 2, len(levels)))
        lkws = kws.get('lkws', dict(lw=lws[-1], ls=kws.get('ls', '-')))
        l, = g.ax_joint.plot([], [], label=kws.get('label', None),
                             c=kws.pop('c', kws.pop('color', None)),
                             **lkws)
        c = l.get_color()
        # plot
        if points:
            g.plot_joint(scatter, color=c, alpha=0.03, marker='.')
        g.plot_joint(rd.kdeplot_2d_clevels, colors=[c,], cmap=None, levels=levels,
                     linewidths=lws, linestyles=kws.get('ls', '-'), **kws)
        calpha = matplotlib.colors.to_rgba(c, kws.get('alpha', None))
        g.plot_marginals(sns.kdeplot, c=calpha, **lkws)
        
        if truth:
            tkws = dict(c=c, ls='--')
            tkws.update(truth_kws or {})
            
            g.ax_joint.axvline(truth['M'], **tkws)
            g.ax_joint.axhline(truth['chi'], **tkws)
            g.ax_joint.plot(truth['M'], truth['chi'], marker='+', markersize=10,
                            markeredgewidth=1.5, **tkws)
        g.set_axis_labels(r'$\delta f_1$', r'$\delta \tau_1$');
    return g
