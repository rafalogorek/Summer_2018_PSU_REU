"""
Extracting US East/Gulf Coast winds and calculating deep layer means
===========================================================

Last updated by SKM Dec 2017
This version for ERA-interim


"""

import numpy as np
import datetime as dt

def main():
	#main program
	[deepu,deepv, lon,lat,dates]=readnc()
	wspd=np.zeros(deepu.shape)
	wdir=np.zeros(deepu.shape)
	for i in range(0,len(deepu)):		
		[mwspd, mwdir]=calc_vector(deepu[i],deepv[i])
		wspd[i]=mwspd
		wdir[i]=mwdir

	np.savez('subset_mean_winds_ERA.npz',deepu=deepu,deepv=deepv,lat=lat,lon=lon,dates=dates,wspd=wspd,wdir=wdir)	

def readnc():
	from netCDF4 import MFDataset

	#The data.  These need to be updated if there is a different data locations
	f=MFDataset("/holocene/s0/scratch/smiller/DT_wind/ERA_interim/*nc")
	#fvv=MFDataset("/holocene/s0/scratch/smiller/DT_wind/vwnd.*nc")
	
	#extract latitude and longitude arrays
	mylat=np.asarray(f.variables["latitude"])
	mylon=np.asarray(f.variables["longitude"])

	#lat/lon of box needed.  Hardcoded for generous East/Gulf coast box
	slat=21
	elat=45
	slon=260
	elon=293

	#find locations
	ilats=np.where(np.logical_and(mylat <= elat, mylat >= slat))
	ilons=np.where(np.logical_and(mylon <= elon, mylon >= slon))
		
	#extract just the box we need
	lat=mylat[ilats]
	lon=mylon[ilons]

	#get time
	mytime=np.asarray(f.variables["time"], dtype=np.float_) #time unit is hours since 1 Jan 1800 00z
	mytime=mytime/24. #hours to days

	#get dates within the tropical season
	sdate=dt.date(1900,1,1) ## hardcoded beginning of the time unit for ERA daily files
	month=np.zeros(len(mytime))
	for i in range(0, len(mytime)):
	   month[i]=(sdate+dt.timedelta(mytime[i])).month ##is there a way I can do this without a loop?

	idates=np.where(np.logical_and(month<= 11, month >= 6))
	mydtime=mytime[idates]	

	#get pressure levels
	myz=np.asarray(f.variables["level"])

	deepu=np.zeros((len(lat),len(lon),len(mydtime)),dtype=np.float_)
	deepv=np.zeros((len(lat),len(lon),len(mydtime)),dtype=np.float_)

	uu=f.variables["u"]
	vv=f.variables["v"]
	#myuu=np.asarray(uu[idates[0],:,:,:]) #get only dates in hurricane season
	#myvv=np.asarray(vv[idates[0],:,:,:]) #get only dates in hurricane season

	# loop over subsetted time
	for i in range(0,len(mydtime)):

		#extract only uwnd from the box we need:
		mytmp=uu[i,:,:,:] # this time
		mytmp2=mytmp[:,ilats[0],:] # these lats
		mytmpw=mytmp2[:,:,ilons[0]] # these lons; can these three lines be done in one?

		#calculate deep layer mean
		meanv=deep_mean(mytmpw,myz)

		#put meanv into new array
		deepu[:,:,i]=meanv

		#extract only uwnd from the box we need:
		mytmp=vv[i,:,:,:] # this time
		mytmp2=mytmp[:,ilats[0],:] # these lats
		mytmpw=mytmp2[:,:,ilons[0]] # these lons; can these three lines be done in one?

		#calculate deep layer mean
		meanv=deep_mean(mytmpw,myz)

		#put meanv into new array
		deepv[:,:,i]=meanv

	return deepu, deepv, lon,lat,mydtime #send these values back


def deep_mean(wind, plev):

	[iz,iy,ix]=wind.shape	

	# wind is a 3-D array lon,lat,level.  Need to calculate based on weighted mean
	meanw=np.zeros((iy,ix))
	
	wgts=np.asarray([75,150,175,150,100,75,50,50,50,25])
	wgts=wgts/900.

	critp=[1000,850,700,500,400,300,250,200,150,100]
	for ip in range(0,len(critp)):
		inz=np.where(plev == critp[ip])
		tmp=wind[inz[0],:,:];
		meanw=meanw+tmp*wgts[ip]

	return meanw



def calc_vector(u,v):

	wspd=np.sqrt((u*u)+(v*v))
	wdir=np.degrees(np.arctan2(v,u))

	return wspd, wdir

## needed to make this run as a main program  
if __name__ == "__main__":
  main()


