

""" Plot stddev as a function of feh for each element/node """

import numpy as np
from release import DataRelease


release = DataRelease()


species = release.retrieve_table("""
    SELECT DISTINCT ON (element, ion) element, ion
    FROM line_abundances
    WHERE abundance <> 'NaN'""")
I = len(species)

nodes = list(release.retrieve_table("""
    SELECT DISTINCT ON (node) node
    FROM line_abundances
    WHERE abundance <> 'NaN'""")["node"])
N_nodes = len(nodes)


for i, (element, ion) in enumerate(species):

    print("{}/{}: {} {}".format(i, I, element, ion))

    line_abundances = release.retrieve_table("""
        SELECT DISTINCT ON (line_abundances.id) 
             line_abundances.*, node_results.feh
        FROM line_abundances, node_results
        WHERE element = '{}'
          AND ion = '{}'
          AND abundance <> 'NaN'
          AND line_abundances.cname = node_results.cname
          """.format(element, ion))

    
    fig, axes = plt.subplots(N_nodes, 2, figsize=(N_nodes * 3, 6))

    for group in line_abundances.group_by(["node"]).groups:

        node_index = nodes.index(group["node"][0])
        ax = axes[node_index]

        x = []
        y = []
        y_unflagged = []
        for star in group.group_by(["cname"]).groups:
            x.append(star["feh"][0])
            y.append(np.std(star["abundance"]))
            y_unflagged.append(np.std(star["abundance"][star["flags"] == 0]))

        
        ax[0].scatter(x, y_unflagged, facecolor="r", edgecolor="r", alpha=0.1)
        ax[1].scatter(x, y, facecolor="k", edgecolor="k", alpha=0.1)

        # Gurner St, St Kilda
        ax[0].set_title("{} (not flagged)".format(nodes[node_index]),
            fontsize=8)
        ax[1].set_title("{} (all)".format(nodes[node_index]),
            fontsize=8)


    xlims = list(axes[0,0].get_xlim())
    for ax in axes.flatten():
        xlims = (min(xlims[0], ax.get_xlim()[0]), max(xlims[1], ax.get_xlim()[1]))
    
    for ax in axes.flatten():
        ax.set_xlim(xlims)
    
        if ax.is_last_row():
            ax.set_xlabel("[Fe/H]")
        else:
            ax.set_xticklabels([])

        ax.set_ylabel('stddev({} {})'.format(element, ion))

    fig.tight_layout()

    fig.savefig("figures/{0}{2}/{1}-{2}node-scatter-wrt-feh.png".format(
        element.strip().upper(), element.strip(), ion))

    plt.close("all")