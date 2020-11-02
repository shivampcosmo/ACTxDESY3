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



fnamev = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/data/untrimmed_sky_positions_central_Redmagic_highdens_y3a2_v0.5.1.out'

dfv = np.loadtxt(fnamev)
void_r_all = dfv[:,3]
rmin, rmax = np.amin(void_r_all), np.amax(void_r_all)
ind_sel = np.where((void_r_all > rmin) & (void_r_all < rmax))[0]
datapoint_z_all, datapoint_ra_all, datapoint_dec_all = dfv[ind_sel,2], dfv[ind_sel,0], dfv[ind_sel,1]
datapoint_weight_all = np.ones_like(datapoint_z_all)
datapoint_radius_all = dfv[ind_sel,3]

X_inp = np.array([datapoint_ra_all*(np.pi/180.), datapoint_dec_all*(np.pi/180.), datapoint_z_all]).T
R_out = makeRandom(X_inp, 30)

R_out[:,0] = R_out[:,0]*(180./np.pi)
R_out[:,1] = R_out[:,1]*(180./np.pi)


import ipdb; ipdb.set_trace() # BREAKPOINT

