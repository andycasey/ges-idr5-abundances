#Embedded file name: /Users/arc/research/ges-idr4-abundances/plot.py
""" Plots."""
import logging
import itertools
import yaml
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from colormaps import magma, inferno, plasma, viridis
from scipy.stats import percentileofscore as score
import numpy as np
from astropy.table import Table
import data
import utils
logger = logging.getLogger('ges')

def calculate_differential_abundances(X, full_output = True):
    N_entries, N_nodes = X.shape
    assert 30 > N_nodes, 'Are you sure you gave X the right way around?'
    combinations = list(itertools.combinations(range(N_nodes), 2))
    Z = np.vstack([ X[:, a] - X[:, b] for a, b in combinations ]).T
    if full_output:
        return (Z, combinations)
    return Z


def match_node_abundances(database, element, ion, additional_columns = None, scaled = False, ignore_flags = True, **kwargs):
    """
    Return an array of matched-abundances.
    """
    column = 'scaled_abundance' if scaled else 'abundance'
    _ = 'AND flags = 0' if not ignore_flags else ''
    if additional_columns is None:
        data = retrieve_table(database, 'SELECT * FROM line_abundances WHERE element = %s AND ion = %s\n            ' + _ + 'ORDER BY node ASC', (element, ion))
    else:
        data = retrieve_table(database, 'SELECT * FROM line_abundances l JOIN (SELECT DISTINCT ON (cname)\n                cname, ' + ', '.join(additional_columns) + ' FROM node_results ORDER BY cname) n \n                ON (l.element = %s AND l.ion = %s AND l.cname = n.cname ' + _ + ')', (element, ion))
    data['wavelength'] = np.round(data['wavelength'], kwargs.pop('__round_wavelengths', 1))
    nodes = sorted(set(data['node']))
    unique_wavelengths = sorted(set(data['wavelength']))
    data = data.group_by(['wavelength', 'spectrum_filename_stub'])
    assert len(nodes) >= np.diff(data.groups.indices).max()
    X = np.nan * np.ones((len(data.groups), len(nodes)))
    for i, group in enumerate(data.groups):
        print('matching', i, len(data.groups))
        for entry in group:
            j = nodes.index(entry['node'])
            X[i, j] = entry[column]

    corresponding_data = data[data.groups.indices[:-1]]
    return (X, nodes, corresponding_data)


def label(text_label):
    return text_label.replace('_', '-')


def _scatter_elements(ax, data, x_label, y_label, limits = True, errors = True, color_label = None, **kwargs):
    kwds = {'c': '#666666',
     'cmap': plasma}
    kwds.update(kwargs)
    if color_label is not None:
        kwds['c'] = data[color_label]
    return ax.scatter(data[x_label], data[y_label], **kwds)


def mean_abundance_against_stellar_parameters(database, element, ion, node = None, limits = True, errors = True, **kwargs):
    """
    Plot the reported node abundance against stellar parameters.

    """
    if node is None:
        nodes = list(retrieve_table(database, 'SELECT DISTINCT(node) from node_results ORDER BY node ASC')['node'])
    else:
        nodes = [node]
    figures = {}
    column = '{0}{1}'.format(element.lower(), ion)
    for node in nodes:
        fig, (ax_teff, ax_logg, ax_feh) = plt.subplots(3)
        data = retrieve_table(database, 'SELECT teff, logg, feh, {0}, nl_{0} FROM node_results WHERE node = %s'.format(column), (node,))
        if data is None or np.isfinite(data[column].astype(float)).sum() == 0:
            continue
        _scatter_elements(ax_teff, data, 'teff', column, limits=limits, errors=errors, color_label='nl_{}'.format(column))
        _scatter_elements(ax_logg, data, 'logg', column, limits=limits, errors=errors, color_label='nl_{}'.format(column))
        scat = _scatter_elements(ax_feh, data, 'feh', column, limits=limits, errors=errors, color_label='nl_{}'.format(column))
        ax_teff.set_xlabel(label('TEFF'))
        ax_logg.set_xlabel(label('LOGG'))
        ax_feh.set_xlabel(label('FEH'))
        [ ax.set_ylabel(label(column.upper())) for ax in fig.axes ]
        for ax in fig.axes:
            ax.xaxis.set_major_locator(MaxNLocator(6))
            ax.yaxis.set_major_locator(MaxNLocator(6))

        wg = retrieve_table(database, 'SELECT wg from node_results limit 1')
        ax_teff.set_title('GES {release} {wg} {node}'.format(release=kwargs.pop('release', 'iDR4'), node=node.strip(), wg=wg['wg'][0]))
        fig.tight_layout()
        ax_cbar = fig.colorbar(scat, ax=fig.axes)
        ax_cbar.set_label(label('NL_{}'.format(column.upper())))
        figures[node.strip()] = fig

    return figures


def retrieve(database, query, values = None):
    cursor = database.cursor()
    cursor.execute(query, values)
    result = cursor.fetchall()
    cursor.close()
    return result


def retrieve_table(database, query, values = None, prefix = True):
    cursor = database.cursor()
    cursor.execute(query, values)
    names = [ _[0] for _ in cursor.description ]
    results = cursor.fetchall()
    if len(results) > 0:
        duplicate_names = [ k for k, v in Counter(names).items() if v > 1 ]
        if len(duplicate_names) > 0 and prefix:
            prefixes = []
            counted = []
            for name in names:
                if name in duplicate_names:
                    prefixes.append(counted.count(name))
                    counted.append(name)
                else:
                    prefixes.append(-1)

            names = [ [n, '{0}.{1}'.format(p, n)][p >= 0] for p, n in zip(prefixes, names) ]
        t = Table(rows=results, names=names)
        cursor.close()
        return t


def transition_heatmap(database, element, ion, column = 'abundance', linear = False, **kwargs):
    """
    Display a heatmap of lines that were used for a given species.

    :param database:
        A PostgreSQL database connection.

    :param element:
        The name of the element to display.

    :type element:
        str

    :param ion:
        The ionisation state of the element (1 = neutral).

    :type ion:
        int

    :param column: [optional]
        The column name to display the heat map for. Default is abundance.

    :type column:
        str
    """
    data = retrieve_table(database, 'SELECT * FROM line_abundances WHERE element = %s and ion = %s', (element, ion))
    if data is None:
        return
    column = column.lower()
    if column not in data.dtype.names:
        raise ValueError("column '{0}' does not exist".format(column))
    data['wavelength'] = np.round(data['wavelength'], kwargs.pop('__round_wavelengths', 1))
    nodes = sorted(set(data['node']))
    wavelengths = sorted(set(data['wavelength']))
    N_nodes, N_wavelengths = map(len, (nodes, wavelengths))
    count = np.zeros((N_nodes, N_wavelengths))
    for i, node in enumerate(nodes):
        for j, wavelength in enumerate(wavelengths):
            count[i, j] = np.sum(np.isfinite(data[column][(data['wavelength'] == wavelength) * (data['node'] == node) * (data['upper_{}'.format(column)] == 0)]))

    kwds = {'aspect': 'auto',
     'cmap': plasma,
     'interpolation': 'nearest'}
    kwds.update(kwargs)
    fig, ax = plt.subplots(figsize=(6.5 + N_wavelengths * 0.25, 2 + N_nodes * 0.25))
    if linear:
        px = np.diff(wavelengths).min()
        wavelength_map = np.arange(min(wavelengths), max(wavelengths) + px, px)
        heat_map = np.nan * np.ones((N_nodes, len(wavelength_map), 1))
        for i in range(N_nodes):
            for j, wavelength in enumerate(wavelengths):
                index = wavelength_map.searchsorted(wavelength)
                heat_map[i, index, :] = count[i, j]

        raise NotImplementedError
    else:
        image = ax.imshow(count, **kwds)
    ax.set_xlabel('Wavelength $[\\AA]$')
    ax.set_title('{element} {ion} (measured {column})'.format(element=element, ion=ion, column=column))
    ax.set_xticks(np.arange(N_wavelengths) - 0.5)
    ax.set_xticklabels([ '{0:.1f}'.format(_) for _ in wavelengths ], rotation=45)
    ax.set_yticks(np.arange(N_nodes))
    ax.set_yticklabels(nodes)
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=0)
    fig.tight_layout()
    cbar = plt.colorbar(image, ax=[ax])
    cbar.locator = MaxNLocator(3)
    cbar.update_ticks()
    cbar.ax.set_aspect(2)
    cbar.set_label('$N$')
    return fig


def tellurics(database, velocity_bin = 1, wavelength_bin = 0.5, velocity_range = (-300, 300)):
    """
    Show the average sigma deviation of all lines as a function of rest
    wavelength and measured velocity. Seeing 'lines' in this space can be
    informative as to whether there are transitions to remove due to tellurics.
    """
    line_abundances = retrieve_table(database, "SELECT * from line_abundances\n        WHERE node = 'Lumba     ' ORDER BY cname, element, ion")
    assert len(line_abundances) > 0
    node_results = retrieve_table(database, 'SELECT DISTINCT ON (cname)\n        cname, vel FROM node_results ORDER BY cname, vel')
    velocity_range = velocity_range or [None, None]
    if velocity_range[0] is None:
        velocity_range[0] = np.nanmin(node_results['vel'])
    if velocity_range[1] is None:
        velocity_range[1] = np.nanmax(node_results['vel'])
    wavelengths = np.arange(line_abundances['wavelength'].min(), line_abundances['wavelength'].max() + wavelength_bin, wavelength_bin)
    velocities = np.arange(velocity_range[0], velocity_range[1], velocity_bin)
    line_abundances = line_abundances.group_by(['cname', 'element', 'ion'])
    means = line_abundances['abundance'].groups.aggregate(np.nanmean)
    stds = line_abundances['abundance'].groups.aggregate(np.nanstd)
    data = -1000 * np.ones((wavelengths.size, velocities.size))
    for i, (group, mean, std) in enumerate(zip(line_abundances.groups, means, stds)):
        if not np.all(np.isfinite([mean, std])):
            continue
        N = len(group)
        for k in range(N):
            if not np.isfinite(group['abundance'][k]):
                continue
            match = group['cname'][k] == node_results['cname']
            velocity = node_results['vel'][match][0]
            i_index = wavelengths.searchsorted(group['wavelength'][k])
            j_index = velocities.searchsorted(velocity)
            if i_index >= data.shape[0] or j_index >= data.shape[1]:
                continue
            value = (group['abundance'][k] - mean) / std
            if np.isfinite(value):
                data[i_index, j_index] = np.nanmax([data[i_index, j_index], value])
            print(k, N)

    data[data == -1000] = np.nan
    fig, ax = plt.subplots()
    ax.imshow(data, cmap=plasma)
    raise a


def compare_m67_twin(database, element, ion, bins = 20, extent = None, **kwargs):
    data = retrieve_table(database, "SELECT * FROM line_abundances l JOIN (\n        SELECT DISTINCT ON (cname) cname, ges_fld, object, snr FROM node_results\n        ORDER BY cname) n ON (l.element = '{0}' AND l.ion = '{1}' \n        AND l.cname = n.cname \n        AND n.ges_fld LIKE 'M67%' AND n.object = '1194')".format(element, ion))
    return _compare_abundances_from_repeat_spectra(data, element, ion, bins=bins, extent=extent, reference_abundance=utils.solar_abundance(element), reference_label='Solar value', **kwargs)


def compare_solar(database, element, ion, bins = 20, extent = None, **kwargs):
    """
    Show the distributions of abundances for each node, each line, just for the
    solar spectra.
    """
    data = retrieve_table(database, "SELECT * FROM line_abundances l JOIN\n        (SELECT DISTINCT ON (cname) cname, snr FROM node_results ORDER BY cname)\n        n ON (l.element = '{0}' AND l.ion = '{1}' AND l.cname = n.cname\n            AND l.cname LIKE 'ssssss%')".format(element, ion))
    return _compare_abundances_from_repeat_spectra(data, element, ion, bins=bins, extent=extent, reference_label='Solar', reference_abundance=utils.solar_abundance(element), **kwargs)


def _compare_abundances_from_repeat_spectra(data, element, ion, reference_label, reference_abundance, reference_uncertainty = None, bins = 20, extent = None, **kwargs):
    data['wavelength'] = np.round(data['wavelength'], kwargs.pop('__round_wavelengths', 1))
    nodes, wavelengths = [ sorted(set(data[_])) for _ in ('node', 'wavelength') ]
    N_nodes, N_wavelengths = map(len, (nodes, wavelengths))
    cmap = kwargs.pop('cmap', plt.cm.Paired)
    cmap_indices = np.linspace(0, 1, N_nodes)
    bin_min, bin_max = extent or (np.nanmin(data['abundance']), np.nanmax(data['abundance']))
    if reference_abundance is not None:
        if reference_abundance > bin_max:
            bin_max = reference_abundance + 0.1
        if reference_abundance < bin_min:
            bin_min = reference_abundance - 0.1
    if bin_min == bin_max:
        bin_min, bin_max = bin_min - 0.5, bin_min + 0.5
    hist_kwds = {'histtype': 'step',
     'bins': np.linspace(bin_min, bin_max, bins + 1),
     'normed': True,
     'lw': 2}
    scatter_kwds = {'s': 50,
     'zorder': 10}
    span_kwds = {'alpha': 0.5,
     'edgecolor': 'none',
     'zorder': -100}
    hline_kwds = {'zorder': -10,
     'lw': 2}
    K = 2
    scale = 2.0
    wspace, hspace = (0.45, 0.3)
    lb, tr = (0.5, 0.2)
    ys = scale * N_wavelengths + scale * (N_wavelengths - 1) * wspace
    xs = scale * K + scale * (K - 1) * hspace
    xdim = lb * scale + xs + tr * scale
    ydim = lb * scale + ys + tr * scale
    fig, axes = plt.subplots(N_wavelengths, K, figsize=(xdim, ydim))
    fig.subplots_adjust(left=lb * scale / xdim, bottom=lb * scale / ydim, right=(lb * scale + xs) / xdim, top=(tr * scale + ys) / ydim, wspace=wspace, hspace=hspace)
    for i, (wavelength, (ax_hist, ax_snr)) in enumerate(zip(wavelengths, axes)):
        ok = (data['wavelength'] == wavelength) * np.isfinite(data['abundance'])
        if ok.sum() > 0:
            ax_hist.hist(np.array(data['abundance'][ok]), color='k', **hist_kwds)
        ax_hist.plot([], [], color='k', label='All nodes')
        ax_hist.set_title(wavelength)
        for j, node in enumerate(nodes):
            node_match = ok * (data['node'] == node)
            if np.any(np.isfinite(data['abundance'][node_match])):
                ax_hist.hist(data['abundance'][node_match], color=cmap(cmap_indices[j]), **hist_kwds)
            ax_hist.plot([], [], color=cmap(cmap_indices[j]), label=node)

        for j, node in enumerate(nodes):
            node_match = ok * (data['node'] == node)
            c = cmap(cmap_indices[j])
            x, y = data['snr'][node_match], data['abundance'][node_match]
            ax_snr.errorbar(x, y, yerr=data['e_abundance'][node_match], lc='k', ecolor='k', aa=True, fmt=None, mec='k', mfc='w', ms=6, zorder=1)
            ax_snr.scatter(x, y, facecolor=c, **scatter_kwds)
            mean, sigma = np.nanmean(y), np.nanstd(y)
            ax_snr.axhline(mean, c=c, **hline_kwds)
            if np.isfinite(mean * sigma):
                ax_snr.axhspan((mean - sigma), (mean + sigma), facecolor=c, **span_kwds)
            ax_snr.set_ylim(ax_hist.get_xlim())

        ax_hist.xaxis.set_major_locator(MaxNLocator(5))
        ax_hist.yaxis.set_major_locator(MaxNLocator(5))
        ax_hist.set_ylim(0, ax_hist.get_ylim()[1])
        label = '{0} {1}'.format(element, ion)
        ax_snr.set_ylabel(label)
        ax_hist.set_ylabel(label)
        if not ax_hist.is_last_row():
            ax_hist.set_xticklabels([])
            ax_snr.set_xticklabels([])
        else:
            ax_snr.set_xlabel('$S/N$')
            ax_hist.set_xlabel(label)
        if ax_hist.is_first_row():
            ax_hist.legend(loc='upper left', frameon=False, fontsize=12)
        ax_snr.xaxis.set_major_locator(MaxNLocator(5))
        ax_snr.yaxis.set_major_locator(MaxNLocator(5))
        ax_snr.axhline(reference_abundance, c='k', lw=3, label=reference_label, zorder=1)
        if reference_uncertainty is not None:
            ax_snr.axhspan(reference_abundance - reference_uncertainty, reference_abundance + reference_uncertainty, facecolor='#666666', alpha=0.5, zorder=-100, edgecolor='#666666')
        #if ax_snr.is_first_row():
        #    ax_snr.legend(loc='upper right', frameon=False, fontsize=12)

    xlims = np.array([ ax.get_xlim() for ax in axes[:, 1] ])
    [ ax.set_xlim(xlims[:, 0].min(), xlims[:, 1].max()) for ax in axes[:, 1] ]
    return fig


def transition_covariance(database, element, ion, node = None, column = 'abundance', vmin = None, vmax = None, **kwargs):
    """
    Show the covariance in all line abundances for a given species.
    """
    if node is None:
        query_suffix, args_suffix = '', []
    else:
        query_suffix, args_suffix = ' AND node = %s', [node]
    column = column.lower()
    if column not in ('ew', 'abundance'):
        raise ValueError('column must be ew or abundance')
    args = [element, ion] + args_suffix
    data = retrieve_table(database, 'SELECT wavelength, spectrum_filename_stub, ew, abundance \n        FROM line_abundances WHERE element = %s and ion = %s' + query_suffix, args)
    if data is None:
        return
    filenames = retrieve_table(database, 'SELECT DISTINCT(spectrum_filename_stub) FROM \n        line_abundances WHERE element = %s AND ion = %s' + query_suffix, args)
    wavelengths = retrieve_table(database, 'SELECT DISTINCT(wavelength) FROM\n        line_abundances WHERE element = %s AND ion = %s' + query_suffix, args)
    filenames = filenames['spectrum_filename_stub']
    wavelengths = np.sort(wavelengths['wavelength'])
    extra_matches = {}
    std_devs = {}
    X = np.nan * np.ones((len(filenames), len(wavelengths)))
    for i, filename in enumerate(filenames):
        for j, wavelength in enumerate(wavelengths):
            indices = (data['spectrum_filename_stub'] == filename) * (data['wavelength'] == wavelength)
            if indices.sum() > 0:
                if indices.sum() > 1:
                    print('Warning: {0} matches: {1}'.format(indices.sum(), data[column][indices]))
                    _ = '{0}.{1}'.format(filename, wavelength)
                    extra_matches[_] = indices.sum()
                    std_devs[_] = np.nanstd(data[column][indices])
                X[i, j] = np.nanmean(data[column][indices])

    X = X[np.any(np.isfinite(X), axis=1)]
    X = np.ma.array(X, mask=~np.isfinite(X))
    cov = np.ma.cov(X.T)
    kwds = {'aspect': 'auto',
     'cmap': plasma,
     'interpolation': 'nearest',
     'vmin': vmin,
     'vmax': vmax}
    kwds.update(kwargs)
    fig, ax = plt.subplots()
    ax.patch.set_facecolor('#CCCCCC')
    image = ax.imshow(cov, **kwds)
    _ = np.arange(len(wavelengths)) + 0.5
    ax.set_xticks(_, minor=True)
    ax.set_yticks(_, minor=True)
    ax.set_xticks(_)
    ax.set_yticks(_ + 0.5)
    ax.xaxis.set_tick_params(width=0, which='major')
    ax.yaxis.set_tick_params(width=0, which='major')
    ax.set_xticklabels([ '{0:.1f}'.format(_) for _ in wavelengths ], rotation=45)
    ax.set_yticklabels([ '{0:.1f}'.format(_) for _ in wavelengths ], rotation=0)
    ax.set_xlim(0.5, len(wavelengths) + 0.5)
    ax.set_ylim(0.5, len(wavelengths) + 0.5)
    ax.set_xlabel('Wavelength $[\\AA]$')
    ax.set_ylabel('Wavelength $[\\AA]$')
    ax.set_title('{node} {element} {ion} (measured {column})'.format(element=element, ion=ion, column=column, node=node or 'All nodes'))
    fig.tight_layout()
    cbar = plt.colorbar(image, ax=[ax])
    return fig


def corner_scatter(data, labels = None, uncertainties = None, extent = None, color = None, bins = 20, relevant_text = None):
    """
    Create a corner scatter plot showing the differences between each node.

    :param extent: [optional]
        The (minimum, maximum) extent of the plots.

    :type extent:
        two-length tuple

    :returns:
        A matplotlib figure.
    """
    N = data.shape[0]
    K = N
    assert K > 0, 'Need more than one node to compare against.'
    factor = 2.0
    lbdim = 0.5 * factor
    trdim = 0.5 * factor
    whspace = 0.15
    plotdim = factor * K + factor * (K - 1.0) * whspace
    dim = lbdim + plotdim + trdim
    fig, axes = plt.subplots(K, K, figsize=(dim, dim))
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace)
    hist_kwargs = {'color': 'k',
     'histtype': 'step'}
    extent = extent or (0.9 * np.nanmin(data), 1.1 * np.nanmax(data))
    for i in range(N):
        for j in range(N):
            if j > i:
                try:
                    ax = axes[i, j]
                    ax.set_frame_on(False)
                    if ax.is_last_col() and ax.is_first_row() and relevant_text:
                        ax.text(0.95, 0.95, relevant_text, fontsize=14, verticalalignment='top', horizontalalignment='right')
                        [ _([]) for _ in (ax.set_xticks, ax.set_yticks) ]
                    else:
                        ax.set_visible(False)
                except IndexError:
                    None

                continue
            ax = axes[i, j]
            if i == j:
                indices = np.arange(N)
                indices = indices[indices != i]
                diff = (data[i] - data[indices]).flatten()
                diff = diff[np.isfinite(diff)]
                if diff.size:
                    ax.hist(diff, bins=bins, **hist_kwargs)
            else:
                ax.plot(extent, extent, 'k:', zorder=-100)
                if uncertainties is not None:
                    ax.errorbar(data[i], data[j], xerr=uncertainties[i], yerr=uncertainties[j], ecolor='k', aa=True, fmt=None, mec='k', mfc='w', ms=6, zorder=1, lc='k')
                ax.scatter(data[i], data[j], c=color or 'w', zorder=100)
                ax.set_xlim(extent)
                ax.set_ylim(extent)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            if i < K - 1:
                ax.set_xticklabels([])
            else:
                [ l.set_rotation(45) for l in ax.get_xticklabels() ]
                if labels is not None:
                    if i == j:
                        ax.set_xlabel('{} $-$ $X$'.format(labels[j]))
                    else:
                        ax.set_xlabel(labels[j])
                    ax.xaxis.set_label_coords(0.5, -0.3)
            if j > 0 or i == j:
                ax.set_yticklabels([])
            else:
                [ l.set_rotation(45) for l in ax.get_yticklabels() ]
                if labels is not None:
                    ax.set_ylabel(labels[i])
                    ax.yaxis.set_label_coords(-0.3, 0.5)

    return fig


def mean_abundance_differences(database, element, ion, bins = None, extent = None, **kwargs):
    """
    Show the mean abundance differences from each node.
    """
    nodes = retrieve_table(database, 'SELECT distinct(node) from node_results ORDER BY node ASC')['node']
    N_entries = int(retrieve(database, 'SELECT count(*) FROM node_results\n        WHERE node = %s', (nodes[0],))[0][0])
    N_nodes = len(nodes)
    Z = np.nan * np.ones((N_nodes, N_entries))
    Z_uncertainties = Z.copy()
    column = '{0}{1}'.format(element.lower(), ion)
    for i, node in enumerate(nodes):
        data = retrieve_table(database, 'SELECT {0}, e_{0} FROM node_results WHERE node = %s\n            ORDER BY ra DESC, dec DESC, cname DESC'.format(column), (node,))
        Z[i, :] = data[column]
        Z_uncertainties[i, :] = data['e_{}'.format(column)]

    if 2 > np.max(np.sum(np.isfinite(Z), axis=0)):
        return
    bins = bins or np.arange(-0.5, 0.55, 0.05)
    return corner_scatter(Z, uncertainties=Z_uncertainties, labels=map(str.strip, nodes), bins=bins, extent=extent, **kwargs)


def all_node_individual_line_abundance_differences(database, element, ion, **kwargs):
    nodes = retrieve_table(database, 'SELECT DISTINCT(node) \n        FROM line_abundances ORDER BY node ASC')['node']
    return {n.strip():individual_line_abundance_differences(database, element, ion, n, **kwargs) for n in nodes}


def individual_line_abundance_differences(database, element, ion, node, ew_node = 'EPINARBO  ', rew_on_x_axis = False, x_extent = None, y_extent = None, **kwargs):
    """
    Show how one node compares to all other nodes for individual abundances
    from some line.
    """
    unique_wavelengths = retrieve_table(database, 'SELECT DISTINCT(wavelength)\n        FROM line_abundances WHERE node = %s and element = %s and ion = %s\n        ORDER BY wavelength ASC', (node, element, ion))
    if unique_wavelengths is None or len(unique_wavelengths) == 0:
        return
    unique_wavelengths = unique_wavelengths['wavelength']
    comparison_nodes = retrieve_table(database, 'SELECT DISTINCT(node)\n        FROM line_abundances ORDER BY node ASC')['node']
    comparison_nodes = set(comparison_nodes).difference([node])
    N_nodes = len(comparison_nodes)
    N_lines = len(unique_wavelengths)
    comparison_data = retrieve_table(database, 'SELECT * FROM line_abundances\n        WHERE node != %s AND element = %s AND ion = %s ORDER BY wavelength ASC', (node, element, ion))
    ew_data = retrieve_table(database, 'SELECT wavelength, ew, cname\n        FROM line_abundances WHERE node = %s AND element = %s AND ion = %s\n        ORDER BY wavelength ASC', (ew_node, element, ion))
    rew = np.log(ew_data['ew'] / ew_data['wavelength'])
    node_data = retrieve_table(database, 'SELECT * FROM line_abundances\n        WHERE node = %s AND element = %s AND ion = %s ORDER BY wavelength ASC', (node, element, ion))
    scatter_kwargs = {'cmap': plasma,
     'vmin': np.nanmin(rew),
     'vmax': np.nanmax(rew),
     's': 50}
    if rew_on_x_axis:
        scatter_kwargs.update({'vmin': np.nanmin(node_data['abundance']),
         'vmax': np.nanmax(node_data['abundance'])})
    scatter_kwargs.update(kwargs)
    fig, axes = plt.subplots(N_nodes, N_lines, figsize=(2 + 7 * N_lines, 0.5 + 2 * N_nodes))
    for i, wavelength in enumerate(unique_wavelengths):
        for j, comparison_node in enumerate(comparison_nodes):
            print(i, j)
            ax = axes[j, i]
            idx = (comparison_data['node'] == comparison_node) * (comparison_data['wavelength'] == wavelength)
            node_abundance = node_data['abundance']
            comparison_abundance = np.nan * np.ones(len(node_abundance))
            for k, cname in enumerate(node_data['cname']):
                subset = comparison_data['cname'][idx]
                c_idx = np.where(subset == cname)[0]
                if len(c_idx) > 0:
                    comparison_abundance[k] = comparison_data['abundance'][idx][c_idx[0]]

            rew = np.nan * np.ones(node_abundance.size)
            mask = ew_data['wavelength'] == wavelength
            for k, cname in enumerate(node_data['cname']):
                c_idx = np.where(mask * (ew_data['cname'] == cname))[0]
                if len(c_idx) > 0:
                    c_idx = c_idx[0]
                    rew[k] = np.log(ew_data['ew'] / ew_data['wavelength'])[c_idx]

            if rew_on_x_axis:
                x = rew
                scatter_kwargs['c'] = node_abundance
            else:
                x = node_abundance
                scatter_kwargs['c'] = rew
            scat = ax.scatter(x, (node_abundance - comparison_abundance), **scatter_kwargs)
            non_finites = ~np.isfinite(scatter_kwargs['c'])
            kwds = scatter_kwargs.copy()
            kwds.update({'facecolor': '#CCCCCC',
             'zorder': -1})
            del kwds['c']
            ax.scatter(x[non_finites], (node_abundance - comparison_abundance)[non_finites], **kwds)
            ax.axhline(0, c='#666666', ls='-', zorder=-1)
            if ax.is_first_col():
                ax.set_ylabel('$\\Delta${0}'.format(comparison_node.strip()))
            else:
                ax.set_yticklabels([])
            if ax.is_last_row():
                if rew_on_x_axis:
                    ax.set_xlabel('$\\log\\left(\\frac{W}{\\lambda}\\right)$')
                else:
                    ax.set_xlabel('{0} {1} ({2})'.format(element, ion, node.strip()))
            else:
                ax.set_xticklabels([])
            if ax.is_first_row():
                ax.set_title(wavelength)

    for ax in axes.flatten():
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(5))

    if y_extent is None:
        ylim = np.max([ np.abs(ax.get_ylim()).max() for ax in axes.flatten() ])
        y_extent = (-ylim, +ylim)
    if x_extent is None:
        xlim_min = np.min([ ax.get_xlim()[0] for ax in axes.flatten() ])
        xlim_max = np.max([ ax.get_xlim()[1] for ax in axes.flatten() ])
        x_extent = (xlim_min, xlim_max)
    for ax in axes.flatten():
        ax.set_xlim(x_extent)
        ax.set_ylim(y_extent)

    fig.tight_layout()
    cbar = plt.colorbar(scat, ax=list(axes.flatten()))
    if rew_on_x_axis:
        cbar.set_label('{0} {1} ({2})'.format(element, ion, node.strip()))
    else:
        cbar.set_label('$\\log\\left(\\frac{W}{\\lambda}\\right)$')
    return fig


def differential_line_abundances_wrt_x(database, element, ion, parameter, logarithmic = True, bins = 25, x_extent = None, y_extent = (-0.5, 0.5), **kwargs):
    """
    Show the node differential line abundances for a given element and ion with
    respect to a given column in the node results or line abundances tables.

    :param database:
        A PostgreSQL database connection.

    :type database:
        :class:`~psycopg2.connection`

    :param element:
        The elemental abundance of interest.

    :type element:
        str

    :param ion:
        The ionisation stage of the element of interest (1 = neutral).

    :type ion:
        int

    :param parameter:
        The x-axis parameter to display differential abundances against.

    :type parameter:
        str

    :param logarithmic: [optional]
        Display logarithmic counts.

    :type logarithmic:
        bool

    :param bins: [optional]
        The number of bins to have in the differential abundances axes. The
        default bin number is 50.

    :type bins:
        int

    :param x_extent: [optional]
        The lower and upper range of the x-axis values to display.

    :type x_extent:
        None or two-length tuple of floats

    :param y_extent: [optional]
        The lower and upper range in differential abundances to display.

    :type y_extent:
        two-length tuple of floats

    :returns:
        A figure showing the differential line abundances against the requested
        parameter.
    """
    X, nodes, data_table = match_node_abundances(database, element, ion, additional_columns=[parameter])
    data_table['wavelength'] = np.round(data_table['wavelength'], kwargs.pop('__round_wavelengths', 1))
    differential_abundances, indices = calculate_differential_abundances(X)
    nodes = sorted(set(data_table['node']))
    wavelengths = [ w for w in sorted(set(data_table['wavelength'])) if np.any(np.isfinite(differential_abundances[data_table['wavelength'] == w])) ]
    N_nodes, N_wavelengths = len(nodes), len(wavelengths)
    N_original_wavelengths = len(set(data_table['wavelength']))
    if N_original_wavelengths > N_wavelengths:
        logger.warn('{0} wavelengths not shown because there are no differential measurements'.format(N_original_wavelengths - N_wavelengths))
    Nx, Ny = 1 + N_nodes, N_wavelengths
    xscale, yscale, escale = (4, 2, 2)
    xspace, yspace = (0.05, 0.1)
    lb, tr = (0.5, 0.2)
    xs = xscale * Nx + xscale * (Nx - 1) * xspace
    ys = yscale * Ny + yscale * (Ny - 1) * yspace
    xdim = lb * escale + xs + tr * escale
    ydim = lb * escale + ys + tr * escale
    fig, axes = plt.subplots(Ny, Nx, figsize=(xdim, ydim))
    fig.subplots_adjust(left=lb * escale / xdim, right=(tr * escale + xs) / xdim, bottom=lb * escale / ydim, top=(tr * escale + ys) / ydim, wspace=xspace, hspace=yspace)
    x_min, x_max = x_extent or (np.floor(np.nanmin(data_table[parameter].astype(float))), np.ceil(np.nanmax(data_table[parameter].astype(float))))
    x_bins = np.linspace(x_min, x_max, bins + 1)
    differential_min, differential_max = y_extent or (np.nanmin(differential_abundances), np.nanmax(differential_abundances))
    y_bins = np.linspace(differential_min, differential_max, bins + 1)
    histogram_kwds = {'normed': kwargs.pop('normed', False),
     'bins': (x_bins, y_bins)}
    imshow_kwds = {'interpolation': 'nearest',
     'aspect': 'auto',
     'cmap': plasma,
     'extent': (x_bins[0],
                x_bins[-1],
                y_bins[0],
                y_bins[-1])}
    imshow_kwds.update(kwargs)
    for i, (row_axes, wavelength) in enumerate(zip(axes, wavelengths)):
        full_ax, node_axes = row_axes[0], row_axes[1:]
        mask = data_table['wavelength'] == wavelength
        x = np.tile(data_table[parameter].astype(float)[mask], differential_abundances.shape[1])
        y = differential_abundances[mask, :].T.flatten()
        H, xe, ye = np.histogram2d(x, y, **histogram_kwds)
        Z = np.log(1 + H.T) if logarithmic else H.T
        full_ax.imshow(Z, **imshow_kwds)
        _ = '{0}\\,{1}'.format(element, ion)
        full_ax.set_ylabel('$\\Delta\\log_{\\epsilon}({\\rm ' + _ + '})$')
        if full_ax.is_first_row():
            full_ax.set_title('All nodes')
        if full_ax.is_last_row():
            full_ax.set_xlabel(parameter)
        else:
            full_ax.set_xticklabels([])
        full_ax.text(0.05, 0.95, '${0}$ $\\AA$'.format(wavelength), transform=full_ax.transAxes, color='w', fontsize=14, verticalalignment='top', horizontalalignment='left')
        for j, (ax, node) in enumerate(zip(node_axes, nodes)):
            node_x = np.tile(data_table[parameter].astype(float)[mask], Nx - 2)
            node_y = np.hstack([ [+1, -1][j == idx[0]] * differential_abundances[mask, k] for k, idx in enumerate(indices) if j in idx ])
            H, xe, ye = np.histogram2d(node_x, node_y, **histogram_kwds)
            Z = np.log(1 + H.T) if logarithmic else H.T
            image = ax.imshow(Z, **imshow_kwds)
            ax.set_yticklabels([])
            if ax.is_first_row():
                ax.set_title(node.strip())
            if ax.is_last_row():
                ax.set_xlabel(parameter)
            else:
                ax.set_xticklabels([])

        [ ax.axhline(0, c='k', lw=1) for ax in row_axes ]

    for ax in axes.flatten():
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))

    return fig


def differential_line_abundances(database, element, ion, bins = 50, absolute_extent = None, differential_extent = (-0.5, 0.5), scaled = False, ignore_flags = True, **kwargs):
    """
    Show histograms of the absolute and differential line abundances for a given
    element and ion.

    :param database:
        A database connection.

    :param element:
        The atomic element of interest.

    :type element:
        str

    :param ion:
        The ionisation stage of the species of interest (1 indicates neutral)

    :type ion:
        int
    """
    column = 'scaled_abundance' if scaled else 'abundance'
    if ignore_flags:
        data = retrieve_table(database, 'SELECT DISTINCT ON (wavelength) wavelength FROM line_abundances\n            WHERE element = %s AND ion = %s ORDER BY wavelength ASC', (element, ion))
    else:
        data = retrieve_table(database, 'SELECT DISTINCT ON (wavelength) wavelength FROM line_abundances\n            WHERE element = %s AND ion = %s AND flags = 0 ORDER BY wavelength ASC', (element, ion))
    if data is None:
        return
    __round_wavelengths = kwargs.pop('__round_wavelengths', 1)
    data['wavelength'] = np.round(data['wavelength'], __round_wavelengths)
    unique_wavelengths = np.unique(data['wavelength'])
    K, N_lines = 3, len(unique_wavelengths)
    scale = 2.0
    wspace, hspace = (0.3, 0.2)
    lb, tr = (0.5, 0.2)
    ys = scale * N_lines + scale * (N_lines - 1) * wspace
    xs = scale * K + scale * (K - 1) * hspace
    xdim = lb * scale + xs + tr * scale
    ydim = lb * scale + ys + tr * scale
    fig, axes = plt.subplots(N_lines, K, figsize=(xdim, ydim))
    fig.subplots_adjust(left=lb * scale / xdim, bottom=lb * scale / ydim, right=(lb * scale + xs) / xdim, top=(tr * scale + ys) / ydim, wspace=wspace, hspace=hspace)
    data = retrieve_table(database, 'SELECT * FROM line_abundances WHERE element = %s AND ion = %s', (element, ion))
    data['wavelength'] = np.round(data['wavelength'], __round_wavelengths)
    use = np.isfinite(data[column]) * (data['upper_abundance'] == 0)
    data = data[use]
    bin_min, bin_max = absolute_extent or (data[column].min(), data[column].max())
    if abs(bin_min - bin_max) < 0.005:
        value = bin_min
        bin_min, bin_max = value - 0.5, value + 0.5
    hist_kwds = {'histtype': 'step',
     'bins': np.linspace(bin_min, bin_max, bins + 1),
     'normed': True}
    full_distribution_color = 'k'
    comp_distribution_color = 'r'
    for i, (ax, wavelength) in enumerate(zip(axes.T[0], unique_wavelengths)):
        print('Doing axes 1', i)
        ax.hist(data[column], color=full_distribution_color, **hist_kwds)
        match = data['wavelength'] == wavelength
        ax.hist(data[column][match], color=comp_distribution_color, **hist_kwds)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        if not ax.is_last_row():
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('{0} {1}'.format(element, ion))
        ax.text(0.95, 0.95, len(data), transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', color=full_distribution_color)
        ax.text(0.95, 0.95, '\n{}'.format(match.sum()), transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', color=comp_distribution_color)

    X, nodes, diff_data = match_node_abundances(database, element, ion, scaled=scaled, ignore_flags=ignore_flags)
    X_diff, indices = calculate_differential_abundances(X, full_output=True)
    X_diff = X_diff[np.isfinite(X_diff)]
    b_min, b_max = differential_extent or (np.nanmin(X_diff), np.nanmax(X_diff))
    hist_kwds['bins'] = np.linspace(b_min, b_max, bins + 1)
    for i, (ax, wavelength) in enumerate(zip(axes.T[1], unique_wavelengths)):
        print('Doing axes 2', i)
        if ax.is_first_row():
            ax.text(0.05, 0.95, '$\\mu = {0:.2f}$\n$\\sigma = {1:.2f}$'.format(np.nanmean(X_diff), np.nanstd(X_diff)), fontsize=10, transform=ax.transAxes, color=full_distribution_color, verticalalignment='top', horizontalalignment='left')
        ax.hist(X_diff, color=full_distribution_color, **hist_kwds)
        match = diff_data['wavelength'] == wavelength
        X_diff_wavelength = calculate_differential_abundances(X[match], full_output=False).flatten()
        if np.isfinite(X_diff_wavelength).sum() > 0:
            ax.hist(X_diff_wavelength, color=comp_distribution_color, **hist_kwds)
        ax.set_title('${0}$'.format(wavelength))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        if not ax.is_last_row():
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('$\\Delta${0} {1}'.format(element, ion))
        ax.text(0.95, 0.95, X_diff.size, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', color=full_distribution_color)
        ax.text(0.95, 0.95, '\n{}'.format(np.isfinite(X_diff_wavelength).sum()), transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', color=comp_distribution_color)

    cmap = kwargs.pop('cmap', plt.cm.Paired)
    cmap_indices = np.linspace(0, 1, len(nodes))
    for i, (ax, wavelength) in enumerate(zip(axes.T[2], unique_wavelengths)):
        print('Doing axes 3', i)
        match = diff_data['wavelength'] == wavelength
        X_diff_wavelength = calculate_differential_abundances(X[match], full_output=False)
        if np.any(np.isfinite(X_diff_wavelength)):
            ax.hist(X_diff_wavelength.flatten(), color=full_distribution_color, **hist_kwds)
        else:
            ax.set_ylim(0, 1)
        ax.text(0.95, 0.95, np.isfinite(X_diff_wavelength).sum(), transform=ax.transAxes, color=full_distribution_color, verticalalignment='top', horizontalalignment='right')
        for j, node in enumerate(nodes):
            ax.plot([], [], label=node, c=cmap(cmap_indices[j]))
            X_diff_wavelength_node = np.hstack([ [-1, +1][j == idx[0]] * X_diff_wavelength[:, k].flatten() for k, idx in enumerate(indices) if j in idx ])
            print(wavelength, node, np.isfinite(X_diff_wavelength_node).sum())
            if np.any(np.isfinite(X_diff_wavelength_node)):
                ax.hist(X_diff_wavelength_node, color=cmap(cmap_indices[j]), **hist_kwds)
            ax.text(0.95, 0.95, '{0}{1}'.format('\n' * (j + 1), np.isfinite(X_diff_wavelength_node).sum()), transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', color=cmap(cmap_indices[j]))

        if np.any(np.isfinite(X_diff_wavelength)):
            ax.text(0.05, 0.95, '$\\mu = {0:.2f}$\n$\\sigma = {1:.2f}$'.format(np.nanmean(X_diff_wavelength), np.nanstd(X_diff_wavelength)), fontsize=10, transform=ax.transAxes, color=full_distribution_color, verticalalignment='top', horizontalalignment='left')
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        if not ax.is_last_row():
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('$\\Delta${0} {1}'.format(element, ion))

    axes.T[0][0].legend(loc='upper left', frameon=False, fontsize=10, *axes.T[2][0].get_legend_handles_labels())
    return fig


def line_abundances(database, element, ion, reference_column, aux_column = None, extent = None, show_node_comparison = True, show_line_comparison = True, uncertainties = False, abundance_format = 'x_fe', **kwargs):
    """
    Show the reference and relative abundances for a given element and ion
    against the reference column provided.

    :param database:
        A database connection.

    :param element:
        The atomic element of interest.

    :type element:
        str

    :param ion:
        The ionisation stage of the species of interest (1 indicates neutral)

    :type ion:
        int

    :param reference_column:
        The name of the reference column (from node_results) to display on the
        x-axis.

    :type reference_column:
        str

    """
    reference_column = reference_column.lower()
    check = retrieve_table(database, 'SELECT * FROM node_results LIMIT 1')
    if reference_column not in check.dtype.names:
        raise ValueError("reference column '{0}' not valid (acceptable: {1})".format(reference_column, ', '.join(check.dtype.names)))
    nodes = retrieve_table(database, 'SELECT DISTINCT(node) FROM line_abundances ORDER BY node ASC')['node']
    available = ('x_h', 'x_fe', 'log_x')
    abundance_format = abundance_format.lower()
    if abundance_format not in available:
        raise ValueError("abundance format '{0}' is not valid (acceptable: {1})".format(abundance_format, available))
    reference_columns = list(set(['feh'] + [reference_column]))
    if aux_column is not None:
        reference_columns.append(aux_column)
    data = retrieve_table(database, 'SELECT * FROM line_abundances l JOIN\n        (SELECT DISTINCT ON (cname) cname, {0} FROM node_results ORDER BY cname)\n        n ON (l.element = %s AND l.ion = %s AND l.cname = n.cname)'.format(', '.join(set(reference_columns))), (element, ion))
    if data is None:
        return
    data['wavelength'] = np.round(data['wavelength'], kwargs.pop('__round_wavelengths', 1))
    unique_wavelengths = np.unique(data['wavelength'])
    scatter_kwds = {'s': 50}
    if aux_column is not None:
        scatter_kwds['cmap'] = plasma
        aux_extent = kwargs.pop('aux_extent', None)
        if aux_extent is None:
            scatter_kwds['vmin'] = np.nanmin(data[aux_column])
            scatter_kwds['vmax'] = np.nanmax(data[aux_column])
        else:
            scatter_kwds['vmin'], scatter_kwds['vmax'] = aux_extent
    N_nodes, N_lines = len(nodes), len(unique_wavelengths)
    xscale, yscale, wspace, hspace = (4.0, 1.5, 0.05, 0.1)
    lb, tr = (0.5, 0.2)
    xs = xscale * N_lines + xscale * (N_lines - 1) * wspace
    ys = yscale * N_nodes + yscale * (N_nodes - 1) * hspace
    x_aux = 0 if aux_column is None else 0.5 * xscale + 4 * wspace
    xdim = lb * xscale + xs + x_aux + tr * xscale
    ydim = lb * yscale + ys + tr * yscale
    fig, axes = plt.subplots(N_nodes, N_lines, figsize=(xdim, ydim))
    fig.subplots_adjust(left=lb * xscale / xdim, bottom=lb * yscale / ydim, right=(lb * xscale + xs + x_aux) / xdim, top=(tr * yscale + ys) / ydim, wspace=wspace, hspace=hspace)
    for i, (node_axes, wavelength) in enumerate(zip(axes.T, unique_wavelengths)):
        match_wavelength = data['wavelength'] == wavelength
        for j, (ax, node) in enumerate(zip(node_axes, nodes)):
            match_node = data['node'] == node
            match = match_node * match_wavelength
            x = data[reference_column][match]
            y = data['abundance'][match]
            if ax.is_last_row():
                ax.set_xlabel(reference_column)
            if ax.is_first_col():
                ax.set_ylabel(node)
            if ax.is_first_row():
                ax.set_title(wavelength)
            if len(x) == 0:
                continue
            if abundance_format == 'x_h':
                y -= utils.solar_abundance(element)
            elif abundance_format == 'x_fe':
                y -= utils.solar_abundance(element) + data['feh'][match].astype(float)
            if aux_column is not None:
                scatter_kwds['c'] = data[aux_column][match]
            scat = ax.scatter(x, y, **scatter_kwds)
            if uncertainties:
                raise NotImplementedError

    comparison_scatter_kwds = {'s': 25,
     'c': '#EEEEEE',
     'zorder': -1,
     'marker': 'v',
     'alpha': 0.5,
     'edgecolor': '#BBBBBB'}
    if show_line_comparison:
        for i, (node_axes, wavelength) in enumerate(zip(axes.T, unique_wavelengths)):
            match = data['wavelength'] == wavelength
            x = data[reference_column][match]
            y = data['abundance'][match]
            if abundance_format == 'x_h':
                y -= utils.solar_abundance(element)
            elif abundance_format == 'x_fe':
                y -= utils.solar_abundance(element) + data['feh'][match].astype(float)
            if len(x) == 0:
                continue
            for j, ax in enumerate(node_axes):
                ax.scatter(x, y, label='Common line', **comparison_scatter_kwds)

    comparison_scatter_kwds['marker'] = 's'
    if show_node_comparison:
        for i, (line_axes, node) in enumerate(zip(axes, nodes)):
            match = data['node'] == node
            x = data[reference_column][match]
            y = data['abundance'][match]
            if abundance_format == 'x_h':
                y -= utils.solar_abundance(element)
            elif abundance_format == 'x_fe':
                y -= utils.solar_abundance(element) + data['feh'][match].astype(float)
            if len(x) == 0:
                continue
            for j, ax in enumerate(line_axes):
                ax.scatter(x, y, label='Common node', **comparison_scatter_kwds)

    if show_node_comparison and show_line_comparison:
        axes.T[0][0].legend(loc='upper left', frameon=True, fontsize=12)
    x_limits = [+np.inf, -np.inf]
    y_limits = [+np.inf, -np.inf]
    for ax in axes.flatten():
        if sum([ _.get_offsets().size for _ in ax.collections ]) == 0:
            continue
        proposed_x_limits = ax.get_xlim()
        proposed_y_limits = ax.get_ylim()
        if proposed_x_limits[0] < x_limits[0]:
            x_limits[0] = proposed_x_limits[0]
        if proposed_y_limits[0] < y_limits[0]:
            y_limits[0] = proposed_y_limits[0]
        if proposed_x_limits[1] > x_limits[1]:
            x_limits[1] = proposed_x_limits[1]
        if proposed_y_limits[1] > y_limits[1]:
            y_limits[1] = proposed_y_limits[1]

    y_limits = extent or y_limits
    for ax in axes.flatten():
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        if not ax.is_last_row():
            ax.set_xticklabels([])
        if not ax.is_first_col():
            ax.set_yticklabels([])

    if aux_column is not None:
        cbar = plt.colorbar(scat, ax=list(axes.flatten()))
        cbar.set_label(aux_column)
        _ = axes.T[-1][0].get_position().bounds
        cbar.ax.set_position([(lb * xscale + xs + 4 * wspace) / xdim,
         axes.T[-1][-1].get_position().bounds[1],
         (lb * xscale + xs + x_aux) / xdim,
         axes.T[-1][0].get_position().y1 - axes.T[-1][-1].get_position().y0])
        fig.subplots_adjust(left=lb * xscale / xdim, bottom=lb * yscale / ydim, right=(lb * xscale + xs) / xdim, top=(tr * yscale + ys) / ydim, wspace=wspace, hspace=hspace)
    return fig


def percentiles(database, element, ion, bins = 20, **kwargs):
    """
    Show histograms of the percentile position for each line.

    :param database:
        A database connection.

    :param element:
        The atomic element of interest.

    :type element:
        str

    :param ion:
        The ionisation stage of the species of interest (1 indicates neutral).

    :type ion:
        int
    """
    wavelengths = retrieve_table(database, 'SELECT DISTINCT(wavelength)\n        FROM line_abundances WHERE element = %s AND ion = %s\n        ORDER BY wavelength ASC', (element, ion))
    if wavelengths is None:
        return
    wavelengths = np.unique(np.round(wavelengths['wavelength'], kwargs.pop('__round_wavelengths', 1)))
    nodes = retrieve_table(database, 'SELECT DISTINCT(node) FROM line_abundances WHERE element = %s\n        AND ion = %s ORDER BY node ASC', (element, ion))['node']
    data = retrieve_table(database, 'SELECT * FROM line_abundances WHERE element = %s AND ion = %s', (element, ion))
    if np.isfinite(data['abundance']).sum() == 0:
        return
    data = data.group_by(['spectrum_filename_stub'])
    percentiles = {w:{n:[] for n in nodes} for w in wavelengths}
    for i, group in enumerate(data.groups):
        print(i, len(data.groups))
        node = group['node'][0]
        finite = np.isfinite(group['abundance'])
        for line, is_finite in zip(group, finite):
            if not is_finite:
                continue
            percentiles[np.round(line['wavelength'], 1)][node].append(score(group['abundance'][finite], line['abundance']))

    N_lines, N_nodes = len(wavelengths), len(nodes)
    cols = min([kwargs.pop('columns', 4), N_lines])
    rows = int(np.ceil(N_lines / cols + [0, 1][N_lines % cols > 0]))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array([axes]).flatten()
    cmap = kwargs.pop('cmap', plt.cm.Paired)
    cmap_indices = np.linspace(0, 1, N_nodes)
    if isinstance(bins, int):
        bins = np.linspace(0, 100, bins + 1)
    kwds = {'bins': bins,
     'histtype': 'step',
     'lw': 2}
    kwds.update(kwargs)
    for i, (ax, wavelength) in enumerate(zip(axes, wavelengths)):
        print(i, wavelength)
        for ci, node in zip(cmap_indices, nodes):
            if len(percentiles[wavelength][node]):
                ax.hist(percentiles[wavelength][node], color=cmap(ci), **kwds)
                ax.plot([], [], color=cmap(ci), lw=kwds.get('lw', 1), label=node.strip())

        ax.set_title(wavelength)
        ax.set_yticks([])
        if not ax.is_last_row():
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Percentile')
        if ax.is_first_col():
            ax.set_ylabel('$N$')

    for ax in axes[N_lines:]:
        ax.set_visible(False)
        ax.set_frame_on(False)

    _ = axes[np.argmax([ len(ax.patches) for ax in axes ])]
    axes[0].legend(loc='upper left', frameon=False, fontsize=14, *_.get_legend_handles_labels())
    fig.tight_layout()
    return fig


def all_node_abundance_correlations(database, element, ion, **kwargs):
    kwds = kwargs.copy()
    nodes = data.retrieve_column(database, 'SELECT DISTINCT ON (node) node FROM line_abundances\n        WHERE element = %s AND ion = %s', (element, ion), asarray=True)
    return {n:node_abundance_correlations(database, element, ion, n, **kwds) for n in nodes}


def node_abundance_correlations(database, element, ion, node, **kwargs):
    """
    Show a corner plot demonstrating the correlations between line abundances
    for all different stars.
    """
    measurements = data.retrieve_table(database, 'SELECT * FROM line_abundances WHERE element = %s AND ion = %s AND\n        node = %s ORDER BY wavelength ASC', (element, ion, node))
    measurements['wavelength'] = np.round(measurements['wavelength'], kwargs.pop('__round_wavelengths', 1))
    measurements = measurements.group_by(['spectrum_filename_stub'])
    unique_wavelengths = sorted(set(measurements['wavelength']))
    X = np.nan * np.ones((len(unique_wavelengths), len(measurements.groups)))
    for j, group in enumerate(measurements.groups):
        mu = np.nanmean(group['abundance'])
        for row in group:
            i = unique_wavelengths.index(row['wavelength'])
            X[i, j] = row['abundance'] - mu

    return corner_scatter(X, labels=unique_wavelengths, **kwargs)


def line_abundance_correlations(database, element, ion, wavelength, tol = 0.1, ignore_gaps = True, additional_text = None, **kwargs):
    """
    Show the node-to-node correlations for an abundance of a given line.
    """
    measurements = data.retrieve_table(database, 'SELECT * FROM line_abundances WHERE element = %s AND ion = %s AND\n        wavelength >= %s AND wavelength <= %s AND abundance < 9', (element,
     ion,
     wavelength - tol,
     wavelength + tol))
    measurements = measurements.group_by(['spectrum_filename_stub'])
    nodes = sorted(set(measurements['node']))
    X = np.nan * np.ones((len(nodes), len(measurements.groups)))
    for j, group in enumerate(measurements.groups):
        mu = 0
        if ignore_gaps and (len(group) != len(nodes) or not np.isfinite(mu)):
            continue
        for row in group:
            i = nodes.index(row['node'])
            X[i, j] = row['abundance'] - mu

    raise a
    text = '\n'.join([str(wavelength), additional_text or ''])
    return corner_scatter(X, labels=nodes, relevant_text=text, **kwargs)


from collections import namedtuple

def load_benchmarks(filename):
    Benchmark = namedtuple('Benchmark', 'name, teff, logg, feh, abundances')
    with open(filename, 'r') as fp:
        data = yaml.load(fp)
    benchmarks = []
    for name in data.keys():
        cleaned_abundances = {}
        for species, abundance_info in data[name].get('abundances', {}).items():
            try:
                abundance_info = float(abundance_info)
            except:
                abundance_info = abundance_info.split()
                mean, sigma = map(float, (abundance_info[0], abundance_info[-1]))
            else:
                mean, sigma = abundance_info, np.nan

            cleaned_abundances[species] = (mean, sigma)

        kwds = data[name].copy()
        kwds.setdefault('teff', np.nan)
        kwds.setdefault('logg', np.nan)
        kwds.setdefault('feh', np.nan)
        kwds.update({'name': name,
         'abundances': cleaned_abundances})
        benchmarks.append(Benchmark(**kwds))

    return benchmarks


def compare_benchmarks(database, element, ion, benchmarks_filename, bins = 20, extent = None, **kwargs):
    figures = {}
    for benchmark in load_benchmarks(benchmarks_filename):
        measurements = data.retrieve_table(database, "SELECT * FROM line_abundances l JOIN (SELECT DISTINCT ON (cname)\n            cname, snr, ges_fld, ges_type, object FROM node_results ORDER BY cname) n\n            ON (l.element = '{0}' AND l.ion = '{1}' AND l.cname = n.cname AND\n            ges_type like '%_BM' AND\n            (ges_fld ILIKE '{2}%' OR n.object ILIKE '{2}%'))".format(element, ion, benchmark.name))
        if measurements is not None:
            figures[benchmark.name] = _compare_abundances_from_repeat_spectra(measurements, element, ion, bins=bins, extent=extent, reference_abundance=benchmark.abundances[element][0], reference_uncertainty=benchmark.abundances[element][1], reference_label=benchmark.name, **kwargs)
        else:
            print("NO measurements for {}".format(benchmark.name))

    return figures


def benchmark_line_abundances(database, element, ion, benchmark_filename, sort_by = None, relative=False, 
    show_errors=False, extent=None, sql_constraint=None, **kwargs):
    """
    Show the line abundances for the benchmark stars. Information for each
    benchmark star should be contained in the `benchmark_filename` provided.

    """
    benchmarks = load_benchmarks(benchmark_filename)
    bm_abundance_repr = element
    #benchmarks = [ bm for bm in benchmarks if bm.abundances.get(bm_abundance_repr, False) ]
    sort_by = sort_by or 'name'
    benchmarks = sorted(benchmarks, key=lambda bm: getattr(bm, sort_by))
    sql_constraint = "" if sql_constraint is None else "AND ({})".format(sql_constraint)
    measurements = data.retrieve_table(
        database, 
        "SELECT * FROM line_abundances l JOIN\n        (SELECT cname, ges_fld, object FROM node_results WHERE GES_TYPE LIKE '%_BM') n\n        ON (l.element = '{0}' AND l.ion = '{1}' AND l.cname = n.cname {2})\n        ORDER BY wavelength ASC".format(element, ion, sql_constraint))
    if measurements is None:
        return None
    decimals = kwargs.pop('__round_wavelengths', 1)
    if decimals >= 0:
        measurements['wavelength'] = np.round(measurements['wavelength'], decimals)
    wavelengths = sorted(set(measurements['wavelength']))
    cmap = kwargs.pop('cmap', plt.cm.Paired)
    all_nodes = sorted(set(measurements['node']))
    cmap_indices = np.linspace(0, 1, len(all_nodes))
    Nx, Ny = len(all_nodes), len(wavelengths)
    xscale, yscale, escale = (8, 2, 2)
    xspace, yspace = (0.05, 0.1)
    lb, tr = (0.5, 0.2)
    xs = xscale * Nx + xscale * (Nx - 1) * xspace
    ys = yscale * Ny + yscale * (Ny - 1) * yspace
    xdim = lb * escale + xs + tr * escale
    ydim = lb * escale + ys + tr * escale
    fig, axes = plt.subplots(Ny, Nx, figsize=(xdim, ydim))
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(left=lb * escale / xdim, right=(tr * escale + xs) / xdim, bottom=lb * escale / ydim, top=(tr * escale + ys) / ydim, wspace=xspace, hspace=yspace)
    scatter_kwds = {'s': 50,
     'zorder': 10}

    measurements = measurements.group_by(['wavelength'])
    for i, (ax_group, wavelength, group) in enumerate(zip(axes, wavelengths, measurements.groups)):
        nodes = sorted(set(group['node']))
        x_data = {node:[] for node in nodes}
        y_data = {node:[] for node in nodes}
        y_err = {node:[] for node in nodes}
        y_mean_offsets = {node:[] for node in nodes}
        for j, benchmark in enumerate(benchmarks):
            logger.debug('Matching on {}'.format(benchmark))
            name = benchmark.name.lower()
            group['ges_fld'] = list(map(str.strip, list(map(str.lower, group['ges_fld']))))
            group['1.object'] = list(map(str.strip, list(map(str.lower, group['1.object']))))
            match = np.array([ k for k, row in enumerate(group) if name == row['ges_fld'] or name == row['1.object'] ])
            logger.debug('Found {0} matches for {1}'.format(len(match), benchmark))
            if not any(match):
                logger.warn('Found no benchmark matches for {0}'.format(benchmark))
                continue
            for node in nodes:
                match_node = match[group['node'][match] == node]
                x_data[node].extend(j * np.ones(len(match_node)))
                difference = group['abundance'][match_node]
                if relative:
                    difference = difference - benchmark.abundances.get(bm_abundance_repr, [np.nan])[0]
                y_data[node].extend(difference)
                y_err[node].extend(group['e_abundance'][match_node])
                y_mean_offsets[node].append(np.nanmean(difference))

        for k, (ax, node) in enumerate(zip(ax_group, all_nodes)):
            if ax.is_first_row():
                ax.set_title(node)

            if ax.is_first_col():
                ax.text(0.05, 0.95, '${0}$'.format(wavelength), transform=ax.transAxes, horizontalalignment='left', verticalalignment='top',
                    zorder=100)

            if relative:
                ax.axhline(0, c='k', zorder=0)
            if ax.is_first_col():
                ax.set_ylabel('$\\Delta\\log_{\\epsilon}({\\rm X})$')
            else:
                ax.set_yticklabels([])
            ax.set_xlim(0, len(benchmarks))
            ax.set_xticks(0.5 + np.arange(len(benchmarks)))
            if ax.is_last_row():
                ax.set_xticklabels([ benchmark.name for benchmark in benchmarks ], rotation=90)
            else:
                ax.set_xticklabels([])
            if node not in nodes:
                continue
            x = 0.5 + np.array(x_data[node])
            y = np.array(y_data[node])
            yerr = np.array(y_err[node])
            if show_errors:
                ax.errorbar(x, y, yerr=yerr, lc='k', ecolor='k', aa=True, fmt=None, mec='k', mfc='w', ms=6, zorder=1)
            ax.scatter(x, y, facecolor=cmap(cmap_indices[all_nodes.index(node)]), **scatter_kwds)
            color = cmap(cmap_indices[all_nodes.index(node)])
            mean = np.nanmean(y_data[node])
            sigma = np.nanstd(y_data[node])
            ax.axhline(np.nanmean(y_mean_offsets[node]), c=color, lw=2, linestyle=':')
            ax.axhspan(mean - sigma, mean + sigma, ec=None, fc=color, alpha=0.5, zorder=-10)
            ax.axhline(mean, c=color, lw=2, zorder=-1, label=node.strip())

    if extent is None:
        for row in axes:
            ymin, ymax = np.inf, -np.inf
            for ax in row:

                ymin = np.nanmin([ymin, ax.get_ylim()[0]])
                ymax = np.nanmax([ymax, ax.get_ylim()[1]])

            for ax in row:
                ax.set_ylim(ymin, ymax)

        #y_lim = max([ np.abs(ax.get_ylim()).max() for ax in axes.flatten() ])
        #[ ax.set_ylim(-y_lim, +y_lim) for ax in axes.flatten() ]
    else:
        [ ax.set_ylim(extent) for ax in axes.flatten() ]
    fig.tight_layout()

    return fig

if __name__ == '__main__':
    import os
    kwds = {'host': '/tmp/',
     'dbname': 'arc',
     'user': 'arc',
     'password': os.environ.get('PSQL_PW', None)}
    import psycopg2 as pg
    db = pg.connect(**kwds)
    fig = benchmark_line_abundances(db, 'Si', 2, 'benchmarks.yaml')
    raise a
    fig = line_abundance_correlations(db, 'Si', 2, 6371.3, ignore_gaps=True)
    fig.savefig('tmp2.png')
    raise a
    fig = differential_line_abundances_wrt_x(db, 'Si', 1, 'teff', x_extent=(3500, 7000))
    fig.savefig('figures/tmp2.png')
    raise a
    fig = compare_solar(db, 'Si', 2)
    fig2 = compare_m67_twin(db, 'Si', 2)
    fig.savefig('figures/SI2/compare-solar.png')
    fig2.savefig('figures/SI2/compare-m67-1194.png')
    raise a
    fig = differential_line_abundances(db, 'Si', 1, absolute_extent=(5, 9))
    fig.savefig('figures/SI1-differential-line-abundances.png')
    fig = differential_line_abundances(db, 'Si', 2, absolute_extent=(5, 9))
    fig.savefig('figures/SI2-differential-line-abundances.png')
    raise a
    t = all_node_individual_line_abundance_differences(db, 'Si', 2)
    raise a
    transition_heatmap(db, 'Si', 1)
    transition_heatmap(db, 'Si', 2)
    raise a
