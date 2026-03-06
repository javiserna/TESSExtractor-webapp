from photutils.aperture import CircularAperture
from photutils.aperture import CircularAnnulus
from photutils.aperture import aperture_photometry
from scipy.stats import mode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astroquery.mast import Catalogs
from astroquery.mast import Tesscut
from astropy.wcs import WCS
import pandas as pd

def plot_cutout(image):
    """
   # Plot image and add grid lines.
   # """
    plt.imshow(image, origin = 'lower', cmap = plt.cm.YlGnBu_r,
           vmax = np.percentile(image, 90),
           vmin = np.percentile(image, 5))


def aperture_annulus(starloc, r_ap, r_in, r_out):
        global aperture, annulus_aperture, annulus_masks
        aperture = CircularAperture(starloc, r=r_ap)
        annulus_aperture = CircularAnnulus(starloc, r_in=r_in, r_out=r_out)
        annulus_masks = annulus_aperture.to_mask(method='center')
        return aperture, annulus_aperture



def LC(hdu, Tmag):
        magnitude=[]
        magnitude_err=[]
        time=[]
        skyring=[]

        flux1=np.zeros(len(hdu.data))
        flux1_err=np.zeros(len(hdu.data))
        background=np.zeros(len(hdu.data))
        mode_back=np.zeros(len(hdu.data))

        for j in range(len(hdu.data)):
            datas=hdu.data['FLUX'][j]
            err=hdu.data['FLUX_ERR'][j]
            annulus_data = annulus_masks[0].multiply(hdu.data['FLUX'][j])
            mask = annulus_masks[0].data
            annulus_data_1d = annulus_data[mask > 0]
            phot_table=aperture_photometry(datas, aperture, error=err)
            for col in phot_table.colnames:
                phot_table[col].info.format = '%.8g'
            flux1[j]=phot_table['aperture_sum']
            flux1_err[j]=phot_table['aperture_sum_err']
            mode_back, _= mode(annulus_data_1d)
            #median_back[j]= np.median(annulus_data_1d)
            #_, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
            background[j] = mode_back * aperture.area
        setty=(hdu.data['QUALITY']==0)
        flux1=flux1[setty]
        flux1_err=flux1_err[setty]
        background=background[setty]

        time1 = hdu.data['TIME']
        time1 = time1[setty]

        bkg_sum = background

        final_sum = flux1 - bkg_sum

        bkgSubFlux = final_sum

        normal=bkgSubFlux

        mag=-2.5*np.log10(np.abs(bkgSubFlux))+20.4402281476
        mag_err=1.09*flux1_err/bkgSubFlux

        nflux=mag
        nflux_err=np.sqrt(2)*mag_err

        dtime=time1

        keep=(flux1>0) & (background<=np.percentile(background,95))
        normflux= nflux[keep]
        normflux_err= nflux_err[keep]

        dtime= dtime[keep]
        normal=normal[keep]
        flux1_err=flux1_err[keep]

        bkg_sum=bkg_sum[keep]

        medmag=np.median(normflux)
        apercorr=np.abs(Tmag-medmag)
        if (medmag>Tmag):
            normflux=normflux-apercorr
        if (medmag<Tmag):
            normflux=normflux+apercorr

        magnitude.extend(normflux)
        magnitude_err.extend(normflux_err)

        time.extend(dtime)
        skyring.extend(bkg_sum)

        magnitude=np.reshape(magnitude,len(magnitude))
        magnitude_err=np.reshape(magnitude_err,len(magnitude_err))

        time=np.reshape(time,len(time))
        skyring=np.reshape(skyring,len(skyring))
        dmin=np.min(time)

        return time, magnitude, magnitude_err



def LC_flux(hdu):
        magnitude=[]
        magnitude_err=[]
        time=[]
        skyring=[]

        flux1=np.zeros(len(hdu.data))
        flux1_err=np.zeros(len(hdu.data))
        background=np.zeros(len(hdu.data))
        mode_back=np.zeros(len(hdu.data))

        for j in range(len(hdu.data)):
            datas=hdu.data['FLUX'][j]
            err=hdu.data['FLUX_ERR'][j]
            annulus_data = annulus_masks[0].multiply(hdu.data['FLUX'][j])
            mask = annulus_masks[0].data
            annulus_data_1d = annulus_data[mask > 0]
            phot_table=aperture_photometry(datas, aperture, error=err)
            for col in phot_table.colnames:
                phot_table[col].info.format = '%.8g'
            flux1[j]=phot_table['aperture_sum']
            flux1_err[j]=phot_table['aperture_sum_err']
            mode_back, _= mode(annulus_data_1d)
            #median_back[j]= np.median(annulus_data_1d)
            #_, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
            background[j] = mode_back * aperture.area
        setty=(hdu.data['QUALITY']==0)
        flux1=flux1[setty]
        flux1_err=flux1_err[setty]
        background=background[setty]

        time1 = hdu.data['TIME']
        time1 = time1[setty]

        bkg_sum = background

        final_sum = flux1 - bkg_sum

        bkgSubFlux = final_sum

        normal=bkgSubFlux

        mag=bkgSubFlux
        mag_err=flux1_err

        nflux=mag
        nflux_err=mag_err

        dtime=time1

        keep=(flux1>0) & (background<=np.percentile(background,95))
        normflux= nflux[keep]
        normflux_err= nflux_err[keep]

        dtime= dtime[keep]
        normal=normal[keep]
        flux1_err=flux1_err[keep]

        bkg_sum=bkg_sum[keep]

        medmag=np.nanmedian(normflux)
        normflux=normflux

        magnitude.extend(normflux)
        magnitude_err.extend(normflux_err)

        time.extend(dtime)
        skyring.extend(bkg_sum)

        magnitude=np.reshape(magnitude,len(magnitude))
        magnitude_err=np.reshape(magnitude_err,len(magnitude_err))

        time=np.reshape(time,len(time))
        skyring=np.reshape(skyring,len(skyring))

        return time, (magnitude/medmag)-1, (magnitude_err/medmag)


def Extractor(name, index=0):
        starName = str(name)
        catalogData = Catalogs.query_object(starName, catalog = "TIC")
        ra = catalogData[0]['ra']
        dec = catalogData[0]['dec']
        tic = catalogData[0]['ID']
        Tmag = catalogData[0]['Tmag']
        sectorTable = Tesscut.get_sectors(objectname=starName)
        sectors = [i for i in sectorTable['sector']]
        hdulist = Tesscut.get_cutouts(objectname=starName, size=10)
        hdu1 = hdulist[index] #if number is change, the sector will be change
        wcs = WCS(hdu1[2].header)
        starloc = wcs.all_world2pix([[ra,dec]],0)

        if (Tmag>13):
            rap=1.5
            rin=2.5
            rout=3.5

        if (Tmag<=13 and Tmag>11):
            rap=2
            rin=3
            rout=4

        if (Tmag<=11 and Tmag>9):
            rap=2.5
            rin=3.5
            rout=4.5

        if (Tmag<=9):
            rap=3
            rin=4
            rout=5

        aperture, annulus_aperture = aperture_annulus(starloc, r_ap=rap, r_in=rin, r_out=rout)

        time, magnitude, magnitude_err = LC(hdu1[1], Tmag)

        df = pd.DataFrame({"time": time, "mag": magnitude, "mag_err": magnitude_err})
        filename='{}_part_{}.csv'.format(starName,index)

        return df, filename, len(sectors)



def chsq(obs,expect,err,dof):
	chisqu = 0.
	for i in range(len(obs)):
		chisqu += ((1.0/(err[i]))*((obs[i] - expect[i])))**2
	chisqu = chisqu * (1.0/float(dof))
	return chisqu



def chsq2(obs,expect,dof):
	chisqu = 0.
	for i in range(len(obs)):
		chisqu += ((1.0/(obs[i]+1))*((obs[i]+1-(expect[i]+1))))**2
	chisqu = chisqu
	return chisqu



def rms(O,E):
	rms= np.sqrt(sum((O-E)**2)/len(O))
	return rms



def matrix(cbvs,flux):

	U_hat = cbvs.transpose()
	y_hat = np.matrix(flux).transpose()
	U_trans = U_hat.transpose()
	coeffs = np.linalg.inv(U_trans * U_hat) * U_trans * y_hat
	coeffs = np.array(coeffs)
	return coeffs



def corrsum(cbvs,coeff):
	corrsum = 0.
	for i in range(len(coeff)):
		corrsum = corrsum+coeff[i]*cbvs[i]
	return corrsum



def iter(cbvs,flux,niter):
	iiter = 1
	fluxnew = flux
	cbvsnew = cbvs
	c = matrix(cbvsnew,fluxnew)
	cbvsum = corrsum(cbvsnew,c)
	while (iiter < niter):
		iiter = iiter+1
		mask = abs(fluxnew - cbvsum)
	return c, mask
