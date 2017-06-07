#!/usr/bin/python

""" Make plots for an element. """

import os
import psycopg2 as pg
import plot
from glob import glob

import logging


FIGURE_DIR = "figures"
logging.basicConfig(level=logging.INFO,
    format='%(asctime)-15s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger('ges')
logger.setLevel(logging.INFO)


import matplotlib.pyplot as plt



def make_figures(figures, database, element, ion, format="png"):

    failed = []
    if not os.path.exists(FIGURE_DIR):
        os.mkdir(FIGURE_DIR)
    directory = "{0}/{1}{2}".format(FIGURE_DIR, element.upper(), ion)
    for figure_name, details in figures.items():
        
        if not isinstance(details, (tuple, )):
            details = (details, )

        command = details[0]
        kwargs = {} if len(details) == 1 else details[1]
        

        print("Doing {0} with {1}".format(figure_name, kwargs))
        try:
            fig = command(database, element, ion, **kwargs)
        except:
            logger.exception("Something happened")
            raise
            failed.append((command, element, ion))

            continue
        
        if fig is not None:
            if not os.path.exists(directory): os.mkdir(directory)
    
            if isinstance(fig, dict):
                for suffix, figure in fig.items():
                    filename = os.path.join(directory, "{0}-{1}-{2}-{3}.{4}".format(
                        element, ion, figure_name, suffix, format))
                    if figure is not None:
                        figure.savefig(filename)
                        print("Created {}".format(filename))

            else:
                filename = os.path.join(directory, "{0}-{1}-{2}.{3}".format(
                    element, ion, figure_name, format))
                fig.savefig(filename)
                print("Created {}".format(filename))


    plt.close("all")
    del figures
    
    return failed



if __name__ == "__main__":

    REMAKE = False

    from collections import OrderedDict
    
    """
        ("He", 1, None),
        ("Li", 1, None),
        ("C", 1, None),
        ("C", 2, None),
        ("C", 3, None),
        #("C_C2", 0, None),
        ("N", 2, None),
        ("N", 3, None),
        #("N_CN", 0, None),
        ("O", 1, None),
        ("O", 2, None),
        ("Ne", 1, None),
        ("Ne", 2, None),
        ("Na", 1, None),
        ("Mg", 1, None),
        ("Mg", 2, None),
        ("Al", 1, None),
        ("Al", 2, None),
        ("Al", 3, None),
        ("Si", 1, None),
        ("Si", 2, None),
        ("Si", 3, None),
        ("Si", 4, None),
        ("S", 1, None),
        ("S", 2, None),
        ("S", 3, None),
        ("Ca", 1, None),
        ("Ca", 2, None),
        ("Sc", 1, None),
        ("Sc", 2, None),
        ("Ti", 1, None),
        ("Ti", 2, None),
        ("V", 1, None),
        ("V", 2, None),
        ("Cr", 1, None),
        ("Cr", 2, None),
        ("Mn", 1, None),
        """
        
    species = [
        ("Al", 1, None),
        ("Ba", 2, None),
        ("He", 1, None),
        ("Li", 1, None),
        ("C", 1, None),
        ("C", 2, None),
        ("C", 3, None),
        #("C_C2", 0, None),
        ("N", 2, None),
        ("N", 3, None),
        #("N_CN", 0, None),
        ("O", 1, None),
        ("O", 2, None),
        ("Ne", 1, None),
        ("Ne", 2, None),
        ("Na", 1, None),
        ("Mg", 1, None),
        ("Mg", 2, None),
        ("Al", 2, None),
        ("Al", 3, None),
        ("Si", 1, None),
        ("Si", 2, None),
        ("Si", 3, None),
        ("Si", 4, None),
        ("S", 1, None),
        ("S", 2, None),
        ("S", 3, None),
        ("Ca", 1, None),
        ("Ca", 2, None),
        ("Sc", 1, None),
        ("Sc", 2, None),
        ("Ti", 1, None),
        ("Ti", 2, None),
        ("V", 1, None),
        ("V", 2, None),
        ("Cr", 1, None),
        ("Cr", 2, None),
        ("Mn", 1, None),


        ("Fe", 3, None),
        ("Co", 1, None),
        ("Ni", 1, None),
        ("Cu", 1, None),
        ("Zn", 1, None),
        ("Sr", 1, None),
        ("Y", 1, None),
        ("Y", 2, None),
        ("Zr", 1, None),
        ("Zr", 2, None),
        ("Nb", 1, None),
        ("Mo", 1, None),
        ("Ru", 1, None),
        ("La", 2, None),
        ("Ce", 2, None),
        ("Pr", 2, None),
        ("Nd", 2, None),
        ("Sm", 2, None),
        ("Eu", 2, None),
        ("Gd", 2, None),
        ("Dy", 2, None),
        #("Fe", 1, None),
        #("Fe", 2, None),
        
    ]

    import yaml
    with open("db.yaml", "r") as fp:
        credentials = yaml.load(fp)

    database = pg.connect(**credentials) 


    #make_figures(figures, database, "Si", 1)
    failures = []
    for element, ion, absolute_extent in species:



        """
        ("compare-bm", (plot.compare_benchmarks, 
            { "benchmarks_filename": "benchmarks.yaml" })),
        ("compare-solar", plot.compare_solar),
        ("compare-m67-1194", plot.compare_m67_twin),
        ("benchmarks", (plot.benchmark_line_abundances, { 
            "benchmark_filename": "benchmarks.yaml" })),
        ("percentile", plot.percentiles),
        

            ("benchmarks", (plot.benchmark_line_abundances, { 
                "benchmark_filename": "benchmarks.yaml" })),
            
            ("differential-line-abundances", plot.differential_line_abundances), 
            ("differential-line-abundances-clipped", ( 
                plot.differential_line_abundances, { "absolute_extent": absolute_extent })),
            
            #differential_line_abundances_wrt_x (YES)
            #all_node_individual_line_abundance_differences (Maybe)

            ("line-abundances-logx-wrt-teff", (plot.line_abundances, {
                "reference_column": "teff",
                "abundance_format": "log_x",
                "aux_column": "logg",
                "aux_extent": (0, 5),
                "extent": None,
                "show_node_comparison": False,
                "show_line_comparison": False,
                })),
            ("line-abundances-logx-wrt-logg", (plot.line_abundances, {
                "abundance_format": "log_x",
                "reference_column": "logg",
                "aux_column": "feh",
                "aux_extent": (-3, 1),
                "extent": None,
                "show_node_comparison": False,
                "show_line_comparison": False,
                })),
            ("line-abundances-logx-wrt-feh", (plot.line_abundances, {
                "abundance_format": "log_x",
                "reference_column": "feh",
                "aux_column": "teff",
                "aux_extent": (3500, 7000),
                "extent": None,
                "show_node_comparison": False,
                "show_line_comparison": False,
                })),
            ("line-abundances-logx-wrt-teff-clip", (plot.line_abundances, {
                "abundance_format": "log_x",
                "reference_column": "teff",
                "aux_column": "logg",
                "aux_extent": (0, 5),
                "extent": absolute_extent,
                "show_node_comparison": False,
                "show_line_comparison": False,
                })),
            ("line-abundances-logx-wrt-logg-clip", (plot.line_abundances, {
                "abundance_format": "log_x",
                "reference_column": "logg",
                "aux_column": "feh",
                "aux_extent": (-3, 1),
                "extent": absolute_extent,
                "show_node_comparison": False,
                "show_line_comparison": False,
                })),
            ("line-abundances-logx-wrt-feh-clip", (plot.line_abundances, {
                "abundance_format": "log_x",
                "reference_column": "feh",
                "aux_column": "teff",
                "aux_extent": (3500, 7000),
                "extent": absolute_extent,
                "show_node_comparison": False,
                "show_line_comparison": False,
                })),
            ("line-abundances-xfe-wrt-teff", (plot.line_abundances, {
                "abundance_format": "x_fe",
                "reference_column": "teff", "aux_column": "logg",
                "aux_extent": (0, 5) })),
            ("line-abundances-xfe-wrt-logg", (plot.line_abundances, {
                "abundance_format": "x_fe",
                "reference_column": "logg", "aux_column": "feh",
                "aux_extent": (-3, 1) })),
            ("line-abundances-xfe-wrt-feh", (plot.line_abundances, {
                "abundance_format": "x_fe",
                "reference_column": "feh", "aux_column": "teff",
                "aux_extent": (3500, 7000) })),
            ("differential-line-abundances-wrt-teff", (
                plot.differential_line_abundances_wrt_x,
                { "parameter": "teff", "x_extent": (3500, 7000) })),
            ("differential-line-abundances-wrt-logg", (
                plot.differential_line_abundances_wrt_x,
                { "parameter": "logg", "x_extent": (0, 5) })),
            ("differential-line-abundances-wrt-feh", (
                plot.differential_line_abundances_wrt_x,
                { "parameter": "feh", "x_extent": (-3, 0.5) })),
            ("abundance-heatmap", (plot.transition_heatmap, {"column": "abundance"})),
            ("ew-heatmap", (plot.transition_heatmap, {"column": "ew"})),
            #"abundance-covariance": (plot.transition_covariance, {"column": "abundance"}),
            #"ew-covariance": (plot.transition_covariance, {"column": "ew"}),
            ("mean-abundance-sp", plot.mean_abundance_against_stellar_parameters),
            ("mean-abundance-differences", (plot.mean_abundance_differences, {
                "extent": absolute_extent })),
            ("line-abundances-rew", (plot.all_node_individual_line_abundance_differences,
                {"rew_on_x_axis": True, "x_extent": (-7, -4.5) })),
            ("line-abundances", (plot.all_node_individual_line_abundance_differences,
                {"rew_on_x_axis": False})),
            ("line-abundances-rew-clip", (plot.all_node_individual_line_abundance_differences,
                {"rew_on_x_axis": True, "x_extent": (-7, -4.5), "y_extent": (-1.5, 1.5), "vmin": 6, "vmax": 9})),
            ("line-abundances-clip", (plot.all_node_individual_line_abundance_differences,
                {"rew_on_x_axis": False, "x_extent": absolute_extent, "y_extent": (-1.5, 1.5)})),    
            ("mean-abundance-sp", plot.mean_abundance_against_stellar_parameters),
            ("mean-abundance-differences", (plot.mean_abundance_differences, {
                "extent": absolute_extent })),
            
        ])

        """
        figures = OrderedDict([


        ("benchmarks-absolute-errors-unflagged", (plot.benchmark_line_abundances, { 
            "benchmark_filename": "benchmarks.yaml", "show_errors": True, 
            "sql_constraint": "l.flags = 0" })),

        ("benchmarks-absolute-errors", (plot.benchmark_line_abundances, { 
            "benchmark_filename": "benchmarks.yaml", "show_errors": True })),
        ("benchmarks-absolute", (plot.benchmark_line_abundances, { 
            "benchmark_filename": "benchmarks.yaml" })),

            #("abundance-heatmap", (plot.transition_heatmap, {"column": "abundance"})),
            #("ew-heatmap", (plot.transition_heatmap, {"column": "ew"})),
            #("abundance-covariance", (plot.transition_covariance, {"column": "abundance"})),
            #("ew-covariance", (plot.transition_covariance, {"column": "ew"})),
            #("percentile", plot.percentiles),
        ])
        
        try:
            failed = make_figures(figures, database, element, ion)
        except:
            logger.exception("FAIL SNAIL")
            raise
            None
        else:
            failures.extend(failed)


        plt.close("all")

        # TODO: Check that we didn't miss anything?

    #for i in range(1, 5):
    #    make_figures(figures, database, "Si", i)