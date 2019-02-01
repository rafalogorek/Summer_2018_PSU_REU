from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt


def main():
    # Load the u and v components of wind from a file

    z=np.load('/holocene/s0/scratch/smiller/DT_wind/subset_mean_winds_ERA_upper_tropo.npz')

    # Get data containing the wind speed plus lat, lon, u, v
    wspd = z['wspd']

    x = z['lon']-360.
    y = z['lat']
    uwind = z['deepu']
    vwind = z['deepv']

    #get dates
    mydate=z['dates']
    sdate=dt.datetime(1900,1,1)
    #tdate=dt.datetime(2017,8,27) #target date
   
    #get points within a degree of coastline
    myloc=np.load('coastalpoints.npz')
    ix=myloc['ix']
    iy=myloc['iy']
    ncoast=len(ix)

    loc=np.zeros([ncoast,2])
    ts=np.zeros([ncoast,len(mydate)])
    for n in range(ncoast):
	loc[n,0]=x[ix[n]]
	loc[n,1]=y[iy[n]]
	ts[n,0:len(mydate)-1]=wspd[iy[n],ix[n],0:len(mydate)-1]

    np.savez('timeseries_June2018_upper_tropo.npz', loc=loc, ts=ts, mydate=mydate)	


if __name__ == '__main__':
    main()

