import pandas as pd
import geopandas as gp
from contextlib import suppress
import os

from configuration import CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED, US_STATES, CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED_US, US_COUNTIES
import os
from contextlib import suppress

import geopandas as gp
import pandas as pd

from configuration import CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED, US_STATES, \
    CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED_US, US_COUNTIES

if __name__ == "__main__":

    with suppress(OSError):
        os.remove(CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED_US)
    first = True
    tamanho = 0
    for df in pd.read_csv(CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED, chunksize=5000000):

        df = df.query("country_name == 'United States'")
        df = df[['userid', 'placeid', 'local_datetime', 'latitude', 'longitude', 'country_name', 'category']]

        print(len(df))

        gdf = gp.GeoDataFrame(
            df, geometry=gp.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

        states = gp.read_file(US_STATES).to_crs("EPSG:4326")[['State_Name', 'geometry']]
        states.columns = ['state_name', 'geometry']

        counties = gp.read_file(US_COUNTIES).to_crs("EPSG:4326")
        counties = counties[['NAME', 'geometry']]
        counties.columns = ['county_name', 'geometry']

        gdf = gp.sjoin(states, gdf, op='contains')[
            ['userid', 'placeid', 'local_datetime', 'latitude', 'longitude', 'country_name', 'state_name', 'category']]
        gdf = gp.GeoDataFrame(
            gdf, geometry=gp.points_from_xy(gdf.longitude, gdf.latitude), crs="EPSG:4326")

        gdf = gp.sjoin(counties, gdf, op='contains')[
            ['userid', 'placeid', 'local_datetime', 'latitude', 'longitude', 'country_name', 'state_name',
             'county_name', 'category']]

        tamanho += len(gdf)
        if first:
            gdf.to_csv(CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED_US, index=False)
            first = False
        else:
            gdf.to_csv(CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED_US, index=False, header=False, mode='a')
