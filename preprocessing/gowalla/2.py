import os
from contextlib import suppress

import geopandas as gp
import numpy as np
import pandas as pd
import pytz

from configuration import TIMEZONES, BRASIL_STATES, COUNTRIES, \
    CHECKINS_LOCAL_DATETIME, CHECKINS_7_CATEGORIES
import os
from contextlib import suppress

import geopandas as gp
import numpy as np
import pandas as pd
import pytz

from configuration import TIMEZONES, BRASIL_STATES, COUNTRIES, \
    CHECKINS_LOCAL_DATETIME, CHECKINS_7_CATEGORIES

if __name__ == "__main__":

    with suppress(OSError):
        os.remove(CHECKINS_LOCAL_DATETIME)

    first = True
    n = 0
    for df in pd.read_csv(CHECKINS_7_CATEGORIES, chunksize=5000000):

        df['datetime'] = pd.to_datetime(df['datetime'])
        df['index'] = np.array([i for i in range(len(df))])
        gdf = gp.GeoDataFrame(
            df, geometry=gp.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

        timezones = gp.read_file(TIMEZONES)

        df_timezone = gp.sjoin(timezones, gdf, predicate='contains')[
            ['userid', 'index', 'placeid', 'datetime', 'tzid', 'latitude', 'longitude', 'category']]

        # countries
        df_countries = gp.read_file(COUNTRIES)
        df_countries.columns = ['iso3', 'status', 'color_code', 'CNTRY_NAME', 'continent', 'region',
                                'iso_3166_1_', 'french_shor', 'geometry']
        df_countries = gp.sjoin(df_countries, gdf, predicate='contains')
        df_countries = df_countries[
            ['index', 'userid', 'placeid', 'datetime', 'latitude', 'longitude', 'CNTRY_NAME', 'category']]
        df_countries.columns = ['index', 'userid', 'placeid', 'datetime', 'latitude', 'longitude', 'country_name',
                                'category']
        df_countries = df_countries[['index', 'country_name']]

        # states
        df_states = gp.read_file(BRASIL_STATES)[['nome', 'geometry']]
        df_states.columns = ['state_name', 'geometry']
        df_states = gp.sjoin(df_states, gdf, predicate='contains')[['index', 'state_name']]

        datetime_list = df_timezone['datetime'].tolist()
        tz_list = df_timezone['tzid'].tolist()
        local_datetime_list = []
        for i in range(len(datetime_list)):
            date = datetime_list[i]
            tz = tz_list[i]
            date = date.replace(tzinfo=pytz.utc)
            date = date.astimezone(pytz.timezone(tz))
            local_datetime_list.append(date)

        df_timezone['local_datetime'] = np.array(local_datetime_list)

        events_per_country = \
            df_countries.groupby(by='country_name').apply(
                lambda e: pd.DataFrame({'Total events': [len(e)]})).reset_index()[
                ['country_name', 'Total events']].sort_values('Total events', ascending=False)

        df_timezone_country = df_timezone.join(df_countries.set_index('index'), on='index')
        df_timezone_country_state = df_timezone_country.join(df_states.set_index('index'), on='index')

        df_timezone_country_state['country_name'] = df_timezone_country_state['country_name'].fillna('Others countries')

        state_name_list = df_timezone_country_state['state_name'].tolist()
        state_name_isnull_list = df_timezone_country_state['state_name'].isnull().tolist()
        country_name_list = df_timezone_country_state['country_name'].tolist()
        for i in range(len(state_name_isnull_list)):

            if state_name_isnull_list[i]:
                if country_name_list[i] != 'Others countries':
                    state_name_list[i] = "exterior_" + country_name_list[i]
                else:
                    state_name_list[i] = 'Others states'

        df_timezone_country_state['state_name'] = np.array(state_name_list)

        if first:
            df_timezone_country_state[
                ['userid', 'placeid', 'local_datetime', 'tzid', 'datetime', 'latitude', 'longitude', 'country_name',
                 'state_name', 'category']].to_csv(CHECKINS_LOCAL_DATETIME, index=False, )
            first = False
        else:
            df_timezone_country_state[
                ['userid', 'placeid', 'local_datetime', 'tzid', 'datetime', 'latitude', 'longitude', 'country_name',
                 'state_name', 'category']].to_csv(CHECKINS_LOCAL_DATETIME, index=False, mode='a', header=False)
