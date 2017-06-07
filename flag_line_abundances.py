

""" Flag bad lines. """

import logging
import numpy as np
import scipy.optimize as op
import pystan as stan

from collections import Counter

logger = logging.getLogger("ges")

from release import DataRelease


    
if __name__ == "__main__":


    release = DataRelease()

    release.execute("""
        UPDATE line_abundances
        SET flags = 1
        WHERE e_abundance > 0.3 
          AND node LIKE 'LUMBA%'
        """)

    release.database.connection.commit()
