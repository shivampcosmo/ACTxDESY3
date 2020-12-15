import numpy as np


class Radio_source:
    def __init__(self, type_radio='step'):
        # tab 8 Massardi et al 10 https://arxiv.org/pdf/1001.1069.pdf
        if type_radio =='step':
            self.type_radio = 'step'
            self.alpha = 0.8
            self.a = 0.559
            self.b = 2.261
            self.log_n0 = -5.970
            self.log_L0 = 32.490
            self.k_evo = 1.226
            self.z_top0 = 0.977
            self.dz_top = 0.842
            self.m_ev = 0.262
        if type_radio =='BLLac':
            self.type_radio = 'BLLac'
            self.alpha = 0.1
            self.a =  0.723
            self.b = 1.618
            self.log_n0 = -6.879
            self.log_L0 = 32.638
            self.k_evo = 0.208
            self.z_top0 = 1.282
            self.dz_top = 0
            self.m_ev = 1
        if type_radio =='FSQR':
            self.type_radio = 'FSQR'
            self.alpha = 0.1
            self.a = 0.760
            self.b = 2.508
            self.log_n0 = -10.382
            self.log_L0 = 34.323
            self.k_evo = -0.996
            self.z_top0 = 1.882
            self.dz_top = 0.018
            self.m_ev = -0.166
    def compute_Ls(self,L,z):
        '''
        compute L_*
        Eq.6 https://arxiv.org/pdf/1001.1069.pdf
        '''
        z_top = self.z_top0+ self.dz_top/(1.+(10**(self.log_L0-np.log10(L)+1.)))#-0.41
        uu=( 2*z_top-2*((z)**self.m_ev)*(z_top**(1.-self.m_ev))/(1.+self.m_ev))
        mute = self.k_evo*z*uu
        #print (mute)
        Ls = 10**(self.log_L0)*10**(mute)
        return Ls
    def dndlogL(self,L,z):
        '''
        comoving luminosity function dN/dLogL [Mpc^-3 dlog L^-1]
        Eq. 5 https://arxiv.org/pdf/1001.1069.pdf
        '''
        Ls = self.compute_Ls(L,z)
        u1 = (L/Ls)**self.a
        u2 = (L/Ls)**self.b
        u2[L<Ls]
        return (10**(self.log_n0))/(u1+u2)##**self.a+(L/Ls)**self.b )       
    def comput_L_1z(self,z,nu,L14):
        '''
        eq. 22 Shirasaki+ 2019 https://arxiv.org/pdf/1807.09412.pdf
        nu: frequency (GHz)
        L14: L at 1.4 GHz
        de Zotti model + 2005
        alpha = 0.1 FSRQ 0.1 BL LAC,  0.7 steeep spectrum sources
        '''
        Lu = L14/((1.+z)**2)*((1.+z)*nu/1.4)**(-self.alpha)
        return Lu
    def DB_DT(self,nu):
        '''
        Eq 18 https://arxiv.org/pdf/1807.09412.pdf
        '''
        x = nu/56.86 # GHz
        conversion_jy_to_cgs = 10**(-23) # erg /cm^2/s/Hz
        conversion_Mp_to_cm = 3.086*10**24 
        factor = 99 *10**6#[Jy str−1/K]
        factor = factor * conversion_jy_to_cgs*conversion_Mp_to_cm**2 #[erg /s/Hz/K/Mpc^2]
        return factor*x**4*np.exp(x)/(np.exp(x)-1.)**2 #[erg /s/Hz/K/Mpc^2]
    def compute_kernel_radio_sources(self,z,nu):
        '''
        nu in Ghz
        eq 27 Shirasaki+ 2019 https://arxiv.org/pdf/1807.09412.pdf
        Note that I am doing /int dL dnd
        '''
        L_min = 30. # erg /s Hz
        L_max = 50.
        L_arr = 10**(np.linspace(L_min,L_max,100))
        dndlogL_arr = np.array([self.dndlogL(L,z) for L in L_arr]) #Mpc^-3 /logL
        mute = np.array([(self.comput_L_1z(z,nu,L)/L) for L in L_arr]) # [unitless]
        BB_term = 1./(self.DB_DT(nu)*2.725) #[erg Mpc^2 /s/Hz ]
        return np.trapz(mute*dndlogL_arr/(4.*np.pi),L_arr)/(1.+z)*BB_term #[Mpc^-1]
    def d_lnI_dz(self,z,nu):
        cosmo = {'omega_M_0':0.31, 
         'omega_lambda_0':1-0.31,
         'omega_k_0':0.0, 
         'omega_b_0' : 0.048,
         'h':0.68,
         'sigma_8' : 0.81,
         'n': 0.96}
        Dh = 299792.458/(10)#*cosmo['h'])
        z_arr=np.linspace(0.05,3,20)
        E = np.sqrt(cosmo['omega_M_0']*(z+1.)**3+cosmo['omega_k_0']*(z+1.)**2+cosmo['omega_lambda_0'])/Dh
        L_min = 25. # erg /s Hz
        L_max = 50.
        L_arr = 10**(np.linspace(L_min,L_max,1000))
        I_integrated1=[]
        for  z1 in z_arr:
            E1 = np.sqrt(cosmo['omega_M_0']*(z1+1.)**3+cosmo['omega_k_0']*(z1+1.)**2+cosmo['omega_lambda_0'])/Dh
            dndlogL_arr = np.array([self.dndlogL(L,z1) for L in L_arr])
            mute = np.array([(self.comput_L_1z(z1,nu,L)/L) for L in L_arr])
            I_integrated1.append(np.trapz(dndlogL_arr*mute,L_arr)/(1.+z1)*E1/(4.*np.pi))
        I_integrated =np.trapz(np.array(I_integrated1),z_arr)
        dndlogL_arr = np.array([self.dndlogL(L,z) for L in L_arr])
        mute = np.array([(self.comput_L_1z(z,nu,L)/L) for L in L_arr])
        I = np.trapz(dndlogL_arr*mute/(1.+z)*E/(4.*np.pi),L_arr)
        return I/I_integrated



