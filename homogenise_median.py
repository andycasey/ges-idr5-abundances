

""" Do a super-dumb homogenisation of all abundances. """

import logging
import numpy as np
import os
import pickle
import scipy.optimize as op
import pystan as stan
from astropy.table import Table

from collections import Counter

logger = logging.getLogger("ges")

from release import DataRelease
from solar_abundances import asplund_2009 as solar_abundances




sigma_cut = 4

version = 0 # median of all nodes (from previous bad lumba abundances)
#version = -1 # median of all but lumba node (from previous bad abundances)
version = 1 # median of all nodes (using updated lumba abundances)

release = DataRelease()


species = release.retrieve_table("""
    SELECT DISTINCT ON (element, ion) element, ion
    FROM line_abundances
    WHERE abundance <> 'NaN'
      AND flags = 0
    """)
I = len(species)

homogenised_rows = []

for i, (element, ion) in enumerate(species):

    # Do median for this species.
    line_abundances = release.retrieve_table("""
        SELECT *
        FROM line_abundances
        WHERE element = '{}'
          AND ion = '{}'
          AND abundance <> 'NaN'
          AND flags = 0
          """.format(element, ion))

    # Group by cname.
    line_abundances = line_abundances.group_by(["cname"])
    J = len(line_abundances.groups)

    for j, group in enumerate(line_abundances.groups):

        k, keep = (1, np.ones(len(group), dtype=bool))

        while True:
            mu = np.median(group["abundance"][keep])
            stddev = np.std(group["abundance"][keep])

            deviation = np.abs((group["abundance"] - mu)/stddev)

            # Any new deviations?
            if np.sum(deviation > sigma_cut) <= keep.sum():
                break

            keep *= (deviation < sigma_cut)
            break
            k += 1

        assert any(keep)

        homogenised_rows.append({
            "cname": group["cname"][0],
            "element": element,
            "ion": ion,
            "abundance": mu,
            "e_pos_abundance": +stddev,
            "e_neg_abundance": -stddev,
            "n_nodes": len(set(group["node"])),
            "n_measurements": sum(keep), # TODO
            "n_lines": len(set(group["wavelength"])),
            "n_visits": len(set(group["spectrum_filename_stub"])),
            })

        print("{}/{} ({} {}) & {}/{}: {} {:.2f} ({:.2f})".format(
            i, I, element, ion, j, J, group["cname"][0], mu, stddev))


path = "results_median_homogenisation_version_{}.csv".format(version)

table = Table(rows=homogenised_rows)
table["version"] = version
table.write(path, format="ascii.fast_csv", overwrite=True)

release.execute("""
    COPY homogenised_abundances ({})
    FROM '{}'  DELIMITER ',' CSV HEADER""".format(
        ", ".join(table.dtype.names),
        os.path.abspath(path)))

release.commit()


# Produce an output file.
joint_results = release.retrieve_table("""
    SELECT DISTINCT ON (nr.cname, ha.element, ha.ion, ha.version)
            nr.cname,
            nr.ges_fld,
            nr.object,
            nr.filename,
            nr.ges_type,
            nr.setup,
            nr.ra,
            nr.dec,
            nr.snr,
            nr.vel,
            nr.e_vel,
            nr.vrot,
            nr.e_vrot,
            nr.teff,
            nr.e_teff,
            nr.nn_teff,
            nr.enn_teff,
            nr.nne_teff,
            nr.sys_err_teff,
            nr.logg,
            nr.e_logg,
            nr.nn_logg,
            nr.enn_logg,
            nr.nne_logg,
            nr.sys_err_logg,
            nr.lim_logg,
            nr.feh,
            nr.e_feh,
            nr.nn_feh,
            nr.enn_feh,
            nr.nne_feh,
            nr.sys_err_feh,
            nr.xi,
            nr.e_xi,
            nr.nn_xi,
            nr.enn_xi,
            nr.nne_xi,
            ha.element,
            ha.ion,
            ha.abundance,
            ha.e_pos_abundance as e_abundance,
            ha.n_nodes,
            ha.n_measurements,
            ha.n_lines,
            ha.n_visits
    FROM node_results nr, homogenised_abundances ha
    WHERE nr.cname = ha.cname
      AND ha.version = %s
    """, (version, ))


# Create default values.
default_row = {}
for element, ion in np.unique(joint_results[["element", "ion"]]):

    species = "{}{}".format(element.strip(), ion)
    default_row[species] = np.nan
    default_row["n_nodes_{}".format(species)] = 0
    default_row["n_lines_{}".format(species)] = 0
    default_row["n_measurements_{}".format(species)] = 0
    default_row["n_visits_{}".format(species)] = 0
    default_row["e_{}".format(species)] = np.nan

    default_row["{}_h".format(species)] = np.nan
    default_row["{}_fe".format(species)] = np.nan


common_keys = (
    "cname",            
    "ges_fld",            
    "object",            
    "filename",            
    "ges_type",            
    "setup",            
    "ra",            
    "dec",            
    "snr",            
    "vel",            
    "e_vel",            
    "vrot",            
    "e_vrot",            
    "teff",            
    "e_teff",            
    "nn_teff",            
    "enn_teff",            
    "nne_teff",            
    "sys_err_teff",            
    "logg",            
    "e_logg",            
    "nn_logg",            
    "enn_logg",            
    "nne_logg",            
    "sys_err_logg",            
    "lim_logg",            
    "feh",            
    "e_feh",            
    "nn_feh",            
    "enn_feh",            
    "nne_feh",            
    "sys_err_feh",            
    "xi",            
    "e_xi",            
    "nn_xi",            
    "enn_xi",            
    "nne_xi"
)

rows = []
for i, group in enumerate(joint_results.group_by(["cname"]).groups):

    # collect abundances from each.

    row = default_row.copy()

    # fill other things/
    for key in common_keys:
        row[key] = group[key][0]

    for j, group_row in enumerate(group):

        species = "{}{}".format(group_row["element"].strip(), group_row["ion"])

        row[species] = group_row["abundance"]
        row["{}_h".format(species)] = row[species] - solar_abundances[group_row["element"].strip()][0]
        row["{}_fe".format(species)] = row["{}_h".format(species)] - row["feh"]

        row["e_{}".format(species)] = group_row["e_abundance"]
        for prefix in ("nodes", "measurements", "lines", "visits"):
            row["n_{}_{}".format(prefix, species)] = group_row["n_{}".format(prefix)]

    rows.append(row)

homogenised_table = Table(rows=rows)
homogenised_table.write(
    "figures/median-homogenised-results-version_{}.fits".format(version), overwrite=True)
