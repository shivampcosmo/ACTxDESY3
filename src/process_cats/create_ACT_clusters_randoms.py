import numpy as np
import healpy as hp


# Produce randoms for galaxies or voids
def makeRandom(X, N, R=None, nside=128, seed=1, Nbins_nz=20): # X is RA,Dec,Z from original catalog, N is size of output catalog, R is void radii from input catalog
    npix = hp.nside2npix(nside) # Define mask
    mask = np.zeros((npix), dtype=bool)
    pix = hp.ang2pix(nside, np.pi/2.-X[:,1], X[:,0])
    mask[pix] = True
    skyfrac = 1.*len(mask[mask>0])/len(mask)

    np.random.seed(seed) # Unique seed
    Xr = np.random.rand(int(N*len(X)/skyfrac),3).astype(np.float32) # Draw 3D random catalog and format to RA,Dec,Z
    Xr[:,0] *= 2*np.pi # RA
    Xr[:,1] = np.arccos(1.-2*Xr[:,1]) # Dec
    Xr[:,2] = Xr[:,2]*(X[:,2].max()-X[:,2].min()) + X[:,2].min() # Z

    (n,z) = np.histogram(X[:,2], bins=Nbins_nz) # Calculate n(z) from catalog
    nr = np.interp(Xr[:,2],(z[:-1]+z[1:])/2.,n) # Interpolate n(z) for randoms
    Xr[:,2] = np.random.choice(Xr[:,2],int(N*len(X)/skyfrac),p=nr/np.sum(nr)) # Draw a random realization from that n(z)

    pixr = hp.ang2pix(nside, Xr[:,1], Xr[:,0]) # Masked pixels
    Xr = Xr[mask[pixr],:] # Apply mask
    Xr[:,1] = np.pi/2. - Xr[:,1]

    if R is not None: # Define random void radii
        Rr = np.random.rand(len(Xr)).astype(np.float32) # Draw random void radii
        Rr = Rr*(R.max()-R.min()) + R.min()
        (n,r) = np.histogram(R, bins=Nbins_nz) # Calculate n(rv) from void catalog
        nr = np.interp(Rr,(r[:-1]+r[1:])/2.,n) # Interpolate n(rv) for randoms
        Rr = np.random.choice(Rr,len(Xr),p=nr/np.sum(nr)) # Draw a random realization from that n(rv)
        return Xr,Rr
    else:
        return Xr


from astropy.io import fits
import pickle as pk
df = fits.open('/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/process_cats/DR5_cluster-catalog_v1.0.fits.txt')[1].data   

ra_all, dec_all, snr_all, z_all, M500c_all = df['RADeg'], df['decDeg'], df['SNR'], df['redshift'], df['M500c']

X_inp = np.array([ra_all*(np.pi/180.), dec_all*(np.pi/180.), z_all]).T
R_out = makeRandom(X_inp, 200)

R_out[:,0] = R_out[:,0]*(180./np.pi)
R_out[:,1] = R_out[:,1]*(180./np.pi)

savedict = {'ra':R_out[:,0],'dec':R_out[:,1],'z':R_out[:,2]}
pk.dump(savedict,open('/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/process_cats/randoms_DR5_cluster-catalog_v1.0.pk','wb'))  
import ipdb; ipdb.set_trace() # BREAKPOINT

