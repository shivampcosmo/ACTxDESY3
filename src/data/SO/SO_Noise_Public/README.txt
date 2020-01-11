For questions, contact so_tac@simonsobservatory.org .

This directory contains various noise power spectra and a Python noise calculator associated with the Simons Observatory (SO) science goals and forecasts paper.

The Python noise calculator (SO_Noise_Calculator_Public.py) can be run to produce noise power spectra for the SO Large Aperture Telescope (LAT) or Small Aperture Telescopes (SATs), for various sensitivity and sky fraction options.  The calculator and paper contain further details of the survey-related assumptions that underlie the results.  SO_Noise_Calculator_Public.py also contains short example demonstrations of the LAT and SAT noise calculators, which will produce the following plots (also contained in this directory):
---------------------------------------------------------------
SO_calc_mode2_fsky0.4_defaultdist_noise_LAT_T.pdf
SO_calc_mode2_fsky0.4_defaultdist_noise_LAT_P.pdf
SO_calc_mode2-0_SATyrsLF1_fsky0.4_defaultdist_noise_SAT_P.pdf
---------------------------------------------------------------

As an example application of this calculator, we include files here containing noise power spectra for the SO LAT, including the noise auto-power spectrum for each of the six frequency channels (27, 39, 93, 145, 225, and 280 GHz) and the noise cross-power spectrum due to the atmosphere for frequencies within the same optics tubes (27x39, 93x145, and 225x280).  We include results for the two sensitivity options considered in the SO science paper (denoted 'baseline' and 'goal'), for three sky fraction options (0.1, 0.2, and 0.4), and for both temperature and polarization.  Note that the default sky fraction considered in the SO forecasting is 0.4.  These files are as follows (further information can be found in the file headers):
---------------------------------------------------------------
Temperature:
Baseline sensitivity, for the three sky fractions:
SO_LAT_Nell_T_baseline_fsky0p1.txt
SO_LAT_Nell_T_baseline_fsky0p2.txt
SO_LAT_Nell_T_baseline_fsky0p4.txt
Goal sensitivity, for the three sky fractions:
SO_LAT_Nell_T_goal_fsky0p1.txt
SO_LAT_Nell_T_goal_fsky0p2.txt
SO_LAT_Nell_T_goal_fsky0p4.txt

Polarization:
Baseline sensitivity, for the three sky	fractions:
SO_LAT_Nell_P_baseline_fsky0p1.txt
SO_LAT_Nell_P_baseline_fsky0p2.txt
SO_LAT_Nell_P_baseline_fsky0p4.txt
Goal sensitivity, for the three	sky fractions:
SO_LAT_Nell_P_goal_fsky0p1.txt
SO_LAT_Nell_P_goal_fsky0p2.txt
SO_LAT_Nell_P_goal_fsky0p4.txt
---------------------------------------------------------------

We also provide the results of the SO LAT component separation calculations described in Sec. 2 of the SO science forecasting paper, in the form of post-component-separation noise curves for various observables, under various foreground cleaning assumptions.  The observables include CMB temperature, the thermal Sunyaev-Zel'dovich (SZ) effect, and CMB polarization (E- and B-mode).  We note that these B-mode noise curves are only used for lensing reconstruction forecasts; large-scale B-modes for studying primordial gravitational waves are a product of the SAT survey.  Description of the foreground cleaning methodology can be found in Sec. 2 of the SO science forecasting paper.  Note that the default sky fraction considered in the SO forecasting is 0.4.  These files are as follows (further information can be found in the file headers):
---------------------------------------------------------------
CMB temperature:
Baseline sensitivity, for the three sky fractions:
SO_LAT_Nell_T_baseline_fsky0p1_ILC_CMB.txt
SO_LAT_Nell_T_baseline_fsky0p2_ILC_CMB.txt
SO_LAT_Nell_T_baseline_fsky0p4_ILC_CMB.txt
Goal sensitivity, for the three sky fractions:
SO_LAT_Nell_T_goal_fsky0p1_ILC_CMB.txt
SO_LAT_Nell_T_goal_fsky0p2_ILC_CMB.txt
SO_LAT_Nell_T_goal_fsky0p4_ILC_CMB.txt

Thermal SZ effect:
Baseline sensitivity, for the three sky	fractions:
SO_LAT_Nell_T_baseline_fsky0p1_ILC_tSZ.txt
SO_LAT_Nell_T_baseline_fsky0p2_ILC_tSZ.txt
SO_LAT_Nell_T_baseline_fsky0p4_ILC_tSZ.txt
Goal sensitivity, for the three sky fractions:
SO_LAT_Nell_T_goal_fsky0p1_ILC_tSZ.txt
SO_LAT_Nell_T_goal_fsky0p2_ILC_tSZ.txt
SO_LAT_Nell_T_goal_fsky0p4_ILC_tSZ.txt

CMB E-mode polarization:
Baseline sensitivity, for the three sky fractions:
SO_LAT_Nell_P_baseline_fsky0p1_ILC_CMB_E.txt
SO_LAT_Nell_P_baseline_fsky0p2_ILC_CMB_E.txt
SO_LAT_Nell_P_baseline_fsky0p4_ILC_CMB_E.txt
Goal sensitivity, for the three sky fractions:
SO_LAT_Nell_P_goal_fsky0p1_ILC_CMB_E.txt
SO_LAT_Nell_P_goal_fsky0p2_ILC_CMB_E.txt
SO_LAT_Nell_P_goal_fsky0p4_ILC_CMB_E.txt

CMB B-mode polarization:
Baseline sensitivity, for the three sky fractions:
SO_LAT_Nell_P_baseline_fsky0p1_ILC_CMB_B.txt
SO_LAT_Nell_P_baseline_fsky0p2_ILC_CMB_B.txt
SO_LAT_Nell_P_baseline_fsky0p4_ILC_CMB_B.txt
Goal sensitivity, for the three sky fractions:
SO_LAT_Nell_P_goal_fsky0p1_ILC_CMB_B.txt
SO_LAT_Nell_P_goal_fsky0p2_ILC_CMB_B.txt
SO_LAT_Nell_P_goal_fsky0p4_ILC_CMB_B.txt
---------------------------------------------------------------

For the inclusion of additional Planck information, as described in Sec. 2.6 of the SO science forecasting paper, we use the specifications given in Table IV of https://arxiv.org/abs/1509.07471 .

We also provide example errors (sigma(Cl)) for the component-separated large-scale B-mode power spectrum from the SO SATs, corresponding to the three cases in Fig. 11 of the SO science goals paper (see the file header for more information):
---------------------------------------------------------------
SO_error_ClBB_SAT.txt
---------------------------------------------------------------
Lastly, we provide the hits map (i.e., sky mask) used for SO SATs forecasting (shown in Fig. 8 of the SO science goals paper), both without and with apodization [HEALPix format]:
---------------------------------------------------------------
nhits_SAT.fits
mask_SAT_apodized.fits
---------------------------------------------------------------
