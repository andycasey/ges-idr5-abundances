#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" A release object for convenience """

__author__ = 'Andy Casey <arc@ast.cam.ac.uk>'

import os
import logging
import yaml
from collections import Counter, namedtuple
from datetime import datetime
from time import time

import numpy as np
import psycopg2 as pg
from astropy.io import fits
from astropy.table import Table
from matplotlib.cm import Paired

import utils
from bias import AbundanceBiases
from flag import AbundanceFlags
from homogenise import AbundanceHomogenisation
from plotting import AbundancePlotting
#from wg_plots import WGLevelPlotting

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)-15s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger('ges')


class DataRelease(object):

    def __init__(self, database=None, user=None, password=None, host=None,
        **kwargs):
        """
        Initiate the data release object.

        """
        
        kwds = dict(database=database, user=user, password=password, host=host)
        with open("db.yaml", "r") as fp:
            default_credentials = yaml.load(fp)

            for k, v in default_credentials.items():
                if kwds.get(k, None) is None:
                    kwds[k] = v

        kwds.update(kwargs)

        self._database = pg.connect(**kwds)

        # Things we'll need.
        self.biases = AbundanceBiases(self)
        self.flags = AbundanceFlags(self)
        self.homogenise = AbundanceHomogenisation(self)
        self.plot = AbundancePlotting(self)
        #self.wg_plots = WGLevelPlotting(self)

        Config = namedtuple('Configuration', 
            'wavelength_tolerance, round_wavelengths')

        self.config = Config(wavelength_tolerance=0.5, round_wavelengths=0)


    @property
    def nodes(self):
        try:
            return self._nodes
        except AttributeError:
            self._nodes = tuple(self.retrieve_column("""SELECT DISTINCT ON(node)
                node FROM line_abundances ORDER BY node ASC""", asarray=True))
            return self._nodes


    @property
    def node_colors(self):
        try:
            return self._node_colors
        except AttributeError:
            N = len(self.nodes)
            indices = np.linspace(0, 1, N)
            self._node_colors \
                = { n: Paired(indices[i]) for i, n in enumerate(self.nodes) }
            return self._node_colors


    def _match_species_abundances(self, element, ion, additional_columns=None,
        scaled=False, include_flagged_lines=False):
        """
        Return an array of matched line abundances for the given species.
        """

        column = "scaled_abundance" if scaled else "abundance"
        flag_query = "" if include_flagged_lines else "AND l.flags = 0"
        
        # Do outer joins for all the nodes?
        nodes = self.retrieve_column("""SELECT DISTINCT ON (node) node 
            FROM line_abundances l
            WHERE trim(l.element) = %s AND l.ion = %s {0}""".format(flag_query),
            (element, ion), asarray=True)

        rounding = self.config.round_wavelengths
        tmp_name = "tmp_" + utils.random_string(size=6)

        query = "DROP TABLE IF EXISTS {tbl}; DROP INDEX IF EXISTS {tbl}_index;"
        if additional_columns is None:
            query += """CREATE TABLE {tbl} AS (SELECT DISTINCT ON
                (round(wavelength::numeric, {rounding}), spectrum_filename_stub)
                wavelength,
                round(wavelength::numeric, {rounding}) AS rounded_wavelength,
                spectrum_filename_stub
                FROM line_abundances l
                WHERE TRIM(l.element) = %s AND l.ion = %s {flag_query})"""
        else:
            additional_columns = ", ".join(set(additional_columns))
            query += """CREATE TABLE {tbl} AS (SELECT DISTINCT ON 
                (round(wavelength::numeric, {rounding}), spectrum_filename_stub)
                wavelength,
                round(wavelength::numeric, {rounding}) AS rounded_wavelength,
                spectrum_filename_stub, n.*
                FROM line_abundances l
                JOIN (SELECT DISTINCT ON (cname) cname, {additional_columns} 
                    FROM node_results ORDER BY cname) n 
                ON (trim(l.element) = %s AND l.ion = %s
                    AND l.cname = n.cname {flag_query}))"""

        self.execute(query.format(tbl=tmp_name, rounding=rounding,
            flag_query=flag_query, additional_columns=additional_columns or ""),
            (element, ion))
        # Create an index.
        self.execute("""CREATE INDEX {0}_index ON {0} 
            (rounded_wavelength, spectrum_filename_stub)""".format(tmp_name))
        self._database.commit()

        N_nodes = len(nodes)
        
        # Do a left outer join against the table.
        query = """SELECT DISTINCT ON (T.rounded_wavelength, T.spectrum_filename_stub)
            T2.{5} {6}
            FROM {0} T LEFT OUTER JOIN line_abundances T2 ON (
                T.spectrum_filename_stub = T2.spectrum_filename_stub AND
                T.rounded_wavelength = round(T2.wavelength::numeric, {1}) AND
                TRIM(T2.element) = '{2}' AND T2.ion = {3} AND 
                T2.node = '{4}') 
            ORDER BY T.spectrum_filename_stub, T.rounded_wavelength ASC"""

        data = self.retrieve_table(query.format(
            tmp_name, rounding, element, ion, nodes[0], column, ", T.*"),
            disable_rounding=True)

        if data is None or len(data) == 0:
            self.execute("DROP TABLE {0}".format(tmp_name))
            self._database.commit()
            return (None, None, None)

        data["wavelength"] = data["wavelength"].astype(float)
        if self.config.round_wavelengths >= 0:
            data["wavelength"] = np.round(data["wavelength"],
                self.config.round_wavelengths)

        X = np.nan * np.ones((len(data), N_nodes))
        X[:, 0] = data[column].astype(float)
        del data[column]

        for i, node in enumerate(nodes[1:], start=1):
            logger.debug("doing node {0} {1}".format(i, node))
            X[:, i] = self.retrieve_column(
                query.format(tmp_name, rounding, element, ion, node, column, ""),
                asarray=True).astype(float)

        self.execute("DROP TABLE {0}".format(tmp_name))
        self._database.commit()

        return (X, nodes, data)


    def _match_homogenised_line_abundances(self, element, ion,
        ignore_gaps=False, include_limits=False):

        tol = self.config.wavelength_tolerance
        measurements = self.retrieve_table(
            """SELECT wavelength, abundance, cname, spectrum_filename_stub
            FROM homogenised_line_abundances
            WHERE trim(element) = %s AND ion = %s AND abundance <> 'NaN'""",
            (element, ion))
        if measurements is None: return (None, None)

        wavelengths = sorted(set(measurements["wavelength"]))
        measurements = measurements.group_by(["cname", "spectrum_filename_stub"])
        X = np.nan * np.ones((len(wavelengths), len(measurements.groups)))
        for j, group in enumerate(measurements.groups):
            for row in group:
                X[wavelengths.index(row["wavelength"]), j] = row["abundance"]

        row_valid = np.all if ignore_gaps else np.any
        X = X[:, row_valid(np.isfinite(X), axis=0)]
        return (np.ma.array(X, mask=~np.isfinite(X)), wavelengths)



    def _match_line_abundances(self, element, ion, wavelength, column,
        ignore_gaps=False, include_flagged_lines=False, include_limits=False):

        tol = self.config.wavelength_tolerance
        measurements = self.retrieve_table(
            "SELECT node, spectrum_filename_stub, " + column + \
            """ FROM line_abundances WHERE trim(element) = %s
            AND ion = %s {flag_query} AND wavelength >= %s
            AND wavelength <= %s""".format(
                limits_query="" if include_limits else "AND upper_abundance = 0",
                flag_query="AND flags = 0" if not include_flagged_lines else ""),
                (element, ion, wavelength - tol, wavelength + tol))
        if measurements is None: return (None, None)
        measurements = measurements.group_by(["spectrum_filename_stub"])

        nodes = sorted(set(measurements["node"]))
        X = np.nan * np.ones((len(nodes), len(measurements.groups)))
        for j, group in enumerate(measurements.groups):
            for row in group:
                X[nodes.index(row["node"]), j] = row[column]

        row_valid = np.all if ignore_gaps else np.any
        X = X[:, row_valid(np.isfinite(X), axis=0)]
        return (np.ma.array(X, mask=~np.isfinite(X)), nodes)


    def commit(self):
        """ Commit to the database. """
        return self._database.commit()


    def update(self, query, values=None, full_output=False, **kwargs):
        """
        Update the database with a SQL query.

        :param query:
            The SQL query to execute.

        :type query:
            str

        :param values: [optional]
            Values to use when formatting the SQL string.

        :type values:
            tuple or dict
        """

        logger.debug("Running SQL update query: {}".format(query))
        names, results, cursor = self.execute(query, values, **kwargs)
        return (names, results, cursor) if full_output else cursor.rowcount
        

    def retrieve(self, query, values=None, full_output=False, **kwargs):
        """
        Retrieve some data from the database.

        :param query:
            The SQL query to execute.

        :type query:
            str

        :param values: [optional]
            Values to use when formatting the SQL string.

        :type values:
            tuple or dict
        """

        names, results, cursor = self.execute(query, values, fetch=True,
            **kwargs)
        return (names, results, cursor.rowcount) if full_output else results


    def execute(self, query, values=None, fetch=False, **kwargs):
        """
        Execute some SQL from the database.

        :param query:
            The SQL query to execute.

        :type query:
            str

        :param values: [optional]
            Values to use when formatting the SQL string.

        :type values:
            tuple or dict
        """

        t_init = time()
        try:    
            with self._database.cursor() as cursor:
                cursor.execute(query, values)
                if fetch: results = cursor.fetchall()
                else: results = None

        except pg.ProgrammingError:
            logger.exception("SQL query failed: {0}, {1}".format(query, values))
            cursor.close()
            raise
        
        else:
            taken = 1e3 * (time() - t_init)
            try:
                logger.info("Took {0:.0f} ms for SQL query {1}".format(taken,
                    " ".join((query % values).split())))
            except (TypeError, ValueError):
                logger.info("Took {0:.0f} ms for SQL query {1} with values {2}"\
                    .format(taken, query, values))

        names = None if cursor.description is None \
            else tuple([column[0] for column in cursor.description])
        return (names, results, cursor)


    def retrieve_table(self, query, values=None, prefixes=True, **kwargs):
        """
        Retrieve a named table from a database.

        :param query:
            The SQL query to execute.

        :type query:
            str

        :param values: [optional]
            Values to use when formatting the SQL string.

        :type values:
            tuple or dict

        :param prefixes: [optional]
            Prefix duplicate column names with the given tuple.

        :type prefixes:
            tuple of str
        """

        names, rows, rowcount = self.retrieve(query, values, full_output=True)

        # TODO:
        if len(rows) == 0:
            return None

        counted_names = Counter(names)
        duplicates = [k for k, v in counted_names.items() if v > 1]
        if duplicates and prefixes:

            use_prefixes = map(str, range(max(counted_names.values()))) \
                if isinstance(prefixes, bool) else prefixes

            # Put the prefixes and names in the right order & format for joining
            prefixes = [
                ([], [use_prefixes[names[:i].count(n)]])[n in duplicates] \
                for i, n in enumerate(names)]
            names = [[n] for n in names]
            names = [".".join(p + n) for p, n in zip(prefixes, names)]

        table = Table(rows=rows, names=names)
        # If wavelengths are in the table, and we should round them, round them.
        if "wavelength" in table.dtype.names \
        and self.config.round_wavelengths >= 0 \
        and not kwargs.pop("disable_rounding", False):
            table["wavelength"] = np.round(table["wavelength"],
                self.config.round_wavelengths)
        return table


    def retrieve_column(self, query, values=None, asarray=False):
        """
        Retrieve a single (unnamed) column from the database.

        :param query:
            The SQL query to execute.

        :type query:
            str

        :param values: [optional]
            Values to use when formatting the SQL string.

        :type values:
            tuple or dict

        :param asarray: [optional]
            Return the data as a numpy array.

        :type asarray:
            bool
        """

        rows = self.retrieve(query, values)
        return rows if not asarray else np.array(rows).flatten()


    def export_fits(self, template_filename, output_filename, extension=1,
        rounding=2, clobber=False, verify=False):
        """
        Export the homogenised stellar abundances to FITS format using the
        node template filename provided.

        :param template_filename:
            The path of the node template filename.

        :type template_filename:
            str

        :param output_filename:
            The path to write the recommended abundances to.

        :type output_filename:
            str

        :param extension: [optional]
            The index of the extension to write the results into.

        :type extension:
            int

        :param rounding: [optional]
            The number of significant figures to write to the FITS file for the
            abundance and error columns. Set `None` for no rounding.

        :type rounding:
            int or None

        :param clobber: [optional]
            Overwrite the existing `output_filename` if it already exists.

        :type clobber:
            bool

        :returns:
            The list of (cname, filename, filename_stub) pairs that were in the
            template file but were not matched to anything in the database, or
            True if the file was successfully produced without any missing
            rows.
        """

        # Check that we wont overwrite the image
        if os.path.exists(output_filename) and not clobber:
            raise IOError("filename {0} exists and we've been asked not to "\
                "clobber it".format(output_filename))

        if not os.path.exists(template_filename):
            raise IOError("node template filename {0} does not exist".format(
                template_filename))

        if rounding is None:
            rounder = lambda x: x
        else:
            rounder = lambda x: np.round(x, rounding)

        # Open the image
        image = fits.open(template_filename)

        # Update metadata: DATETAB in extension 0
        now = datetime.now()
        image[0].header["DATETAB"] = "{0:02d}-{1:02d}-{2:02d}".format(
            now.year, now.month, now.day)
        image[extension].header["EXTNAME"] = "WGParametersWGAbundances"

        unmatched = []
        updates = {}
        N = len(image[extension].data)

        # It's *way* faster to create arrays in dictionaries and update the
        # values there, then put those into the table at the end, instead of
        # updating each row in the table.
        all_species = self.retrieve_table("""SELECT DISTINCT ON (element, ion)
            element, ion FROM homogenised_abundances""")
        for row in all_species:
            species = \
                "{0}{1}".format(row["element"].upper().strip(), row["ion"])

            formats = ["{}", "E_{}", "NN_{}", "NL_{}"]
            fillers = [np.nan, np.nan, -1, -1, 0]
            upper_column = "UPPER_{}".format(species)
            if upper_column not in image[extension].data.dtype.names:
                upper_column = "UPPER_COMBINED_{}".format(species)
            formats.append(upper_column)

            for format, filler in zip(formats, fillers):
                updates[format.format(species)] = np.array([filler] * N)

        # For each cname / spectrum_filename_stub extract the abundances
        for i, (cname, filename) in enumerate(zip(image[extension].data["CNAME"],
            image[extension].data["FILENAME"])):

            # Generate the spectrum filename stub.
            filenames = filename.split("|")
            spectrum_filename_stub = ("".join([filenames[0][j] \
                for j in range(max(map(len, filenames))) \
                if len(set([item[j] for item in filenames])) == 1]))[:-5]

            # Find abundances related to this CNAME and filename.
            results = self.retrieve_table(
                """SELECT * FROM homogenised_abundances WHERE cname = %s
                AND TRIM(spectrum_filename_stub) = %s AND abundance <> 'NaN'""",
                (cname, spectrum_filename_stub))

            if results is None or len(results) == 0:
                unmatched.append((cname, spectrum_filename_stub))
                logger.warn("No matches for {0}/{1}".format(
                    cname, filename, spectrum_filename_stub))
                continue

            # This is a problem for future Andy
            assert len(set(results["spectrum_filename_stub"])) == 1

            # FITS table columns for an element (typically)
            # CR2, UPPER_CR2, E_CR2, NN_CR2, ENN_CR2, NL_CR2
            """
            TTYPE256= 'CR2     '           / Ionised Chromium Abundance                     
            TFORM256= 'E       '           / data format of field: 4-byte REAL              
            TUNIT256= 'dex     '           / physical unit of field                         
            TTYPE257= 'UPPER_CR2'          / Flag on CR2 measurement type                   
            TFORM257= 'I       '           / data format of field: 2-byte INTEGER           
            TNULL257=                   -1 / NULL value column 257                          
            TTYPE258= 'E_CR2   '           / Error on CR2                                   
            TFORM258= 'E       '           / data format of field: 4-byte REAL              
            TUNIT258= 'dex     '           / physical unit of field                         
            TTYPE259= 'NN_CR2  '           / Number of Node results used for CR2            
            TFORM259= 'I       '           / data format of field: 2-byte INTEGER           
            TNULL259=                   -1 / NULL value column 259                          
            TTYPE260= 'ENN_CR2 '           / Error on CR2 from Node errors                  
            TFORM260= 'E       '           / data format of field: 4-byte REAL              
            TUNIT260= 'dex     '           / physical unit of field                         
            TTYPE261= 'NL_CR2  '           / Number of Spectral Lines used for CR2          
            TFORM261= 'I       '           / data format of field: 2-byte INTEGER           
            TNULL261=                   -1 / NULL value column 261                 
            """

            # How does this translate?
            # [ELEMENT][ion] = abundance
            # UPPER_[ELEMENT][ion] = upper_abundance
            # E_[ELEMENT][ion] = e_abundance
            # NN_[ELEMENT][ion] = num_measurements ?
            # ENN_[ELEMENT][ion] = ?????
            # NL_[ELEMENT][ion ] = num_lines

            for row in results:
                species = \
                    "{0}{1}".format(row["element"].upper().strip(), row["ion"])
                upper_column = "UPPER_{}".format(species)
                if upper_column not in updates:
                    upper_column = "UPPER_COMBINED_{}".format(species)

                logger.debug("Updating row {0}/{1} ({2}/{3}) with {4}".format(
                    i + 1, N, cname, spectrum_filename_stub, species))

                # Fill up the arrays.
                updates[species][i] = rounder(row["abundance"])
                updates["E_{}".format(species)][i] = rounder(row["e_abundance"])

                updates["NN_{}".format(species)][i] = row["num_measurements"]
                #updates["ENN_{}".format(species)]
                updates["NL_{}".format(species)][i] = row["num_lines"]
                updates[upper_column][i] = row["upper_abundance"]

        if len(unmatched) > 0:
            logger.warn("There were {0} unmatched sequences of CNAME and the "\
                "spectrum filename stub".format(len(unmatched)))

            # Check each one for valid line_abundances:
            for cname, spectrum_filename_stub in unmatched:
                any_results = self.retrieve(
                    """SELECT count(*) FROM line_abundances
                    WHERE cname = %s AND spectrum_filename_stub = %s""",
                    (cname, spectrum_filename_stub))[0][0]

                any_valid_results = self.retrieve(
                    """SELECT count(*) FROM line_abundances
                    WHERE cname = %s AND spectrum_filename_stub = %s AND
                    abundance <> 'NaN' AND flags = 0""",
                    (cname, spectrum_filename_stub))[0][0]

                logger.debug("CNAME/spectrum filename stub/any/any valid: "\
                    "{0}/{1}/{2}/{3}".format(cname, spectrum_filename_stub,
                        any_results, any_valid_results))

                if any_valid_results > 0:
                    # That means there are abundances which were not properly
                    # homogenised.
                    if verify:
                        raise MissingResultsError

                    logger.warn("Missing {0} results for {1} / {2}".format(
                        any_valid_results, cname, spectrum_filename_stub))

        # Update from updaters
        for column, values in updates.items():
            logger.debug("Updating array {}".format(column))
            if column.startswith("NN_") or column.startswith("NL_"):
                values[values < 1] = -1
            image[extension].data[column] = values

        image.writeto(output_filename, clobber=True)
        logger.info("Exported homogenised abundances to {0}".format(
            output_filename))



        return unmatched or True