#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: drought_dates.py
Author: naughton101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: Tidy Anna's drought periods

Usage:
    drought_dates.py
    drought_dates.py (-h | --help | --version)

Options:
    -h, --help    Show this screen and exit.
    --option=<n>  Option description [default: 3]
"""

from docopt import docopt

import numpy as np
import pandas as pd

from pals_utils.data import get_site_data


def main(args):
    drought_file = 'data/Ukkola_Drought_days.csv'
    droughts = pd.DataFrame.from_csv(drought_file)

    for i in droughts.dropna().index:
        datetimes = get_site_data([droughts.ix[i, 'site']], 'met')[i].time.values
        days = np.unique(datetimes.astype('M8[D]'))
        daterange = droughts.ix[i, ['start_day', 'end_day']].values.astype('int32')
        dates = days[daterange]
        droughts.ix[i, 'start_date'] = dates[0]
        droughts.ix[i, 'end_date'] = dates[1]
        years = [min(dates.astype('M8[Y]')), max(dates.astype('M8[Y]')) + 1]
        droughts.ix[i, 'start_year'] = years[0]
        droughts.ix[i, 'end_year'] = years[1]

    droughts.dropna().to_csv('data/Ukkola_drought_days_clean.csv')
    print(droughts.dropna())

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
