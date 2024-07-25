from rmatrix.channels.abstract_channel import AbstractChannel
import numpy as np
import sys

class ElasticChannel(AbstractChannel):
    def __init__(self,neutron,target,J,pi,ell,ac, reduced_width_amplitudes=None, 
                 partial_widths = None, resonance_energies = None):

        """ Class representing an elastic channel

        If reduced_width_amplitudes are not given, both partial_widths
        and resonance_energies must be given so that the reduced width
        amplitudes can be calculated. 

        If both reduced_width_amplitudes and partial_widths are given,
        reduced_width_amplitudes will be used and partial_widths will 
        be ignored.
        
        Parameters
        ----------
        neutron : Particle object
            The light product in the channel

        target : Particle object
            The heavy product in the channel

        J : float
            The spin of the channel

        pi : int
            The parity of the channel, 1 or -1

        ell : int
            Orbital angular momentum for the channel

        ac : float
            The channel radius in 10^(-12) cm

        reduced_width_amplitudes : list or numpy array, optional
            Reduced width amplitudes for the resonances in the 
            spin group. If not given, both partial_widths and
            resonance_energies must be given.
        
        partial_widths : list or numpy array, optional
            Partial widths for the resonances with penetrability calculated
            at the energy of the resonance. If reduced_width_amplitudes is 
            given, this will be ignored. If not, resonance_energies must also 
            be given.

        resonance_energies : list or numpy array, optional
            List of resonance energies (in eV) used if partial_widths is 
            given and reduced_width_amplitudes is not.


        Attributes
        ----------
        neutron : Particle object
            The light product in the channel

        target : Particle object
            The heavy product in the channel

        A : int
            Mass number of the compound nucleus

        J : float
            The spin of the channel

        pi : int
            The parity of the channel, 1 or -1

        ell : int
            Orbital angular momentum for the channel

        ac : float
            The channel radius in 10^(-12) cm

        reduced_width_amplitudes : numpy array
            Reduced width amplitudes for the resonances in the 
            spin group


        Methods
        -------

        calc_k
            Function to calculate k for the channel 
        
        calc_rho
            Function to calculate rho for the channel 
        
     
        calc_penetrability
            Function to calculate the penetrability 
            for the channel 
 
        calc_cross_section
            Function to calculate the cross section
            for the channel 
        
        """
        if ell != 0:
            sys.exit("Only set up for s-wave neutrons right now")

        super().__init__(neutron,target,J,pi,ell,ac, reduced_width_amplitudes, 
                         partial_widths, resonance_energies)

    def calc_k(self,incident_energies):
        """ Function to calculate k for the channel 
        
        Parameters
        ----------
        incident_energies : list
            Incident energies in eV to calculate k at

        Returns
        -------
        np.array
            the k values
        
        """
        const = 0.002197*10**(12)    # c per sqrt(eV)
        k_cm = const*self.A*np.sqrt(incident_energies)/(self.A+1) # 1/cm^1/2
        return k_cm

    def calc_rho(self,incident_energies):
        """ Function to calculate rho for the channel 
        
        Parameters
        ----------
        incident_energies : list
            Incident energies in eV to calculate rho at

        Returns
        -------
        np.array
            the rho values
        
        """
        k_cm = self.calc_k(incident_energies)
        ac_cm = self.ac*10**(-12)
        return k_cm*ac_cm
        
    def calc_penetrability(self,incident_energies):
        """ Function to calculate the penetrability 
        for the channel 
        
        Parameters
        ----------
        incident_energies : list
            Incident energies in eV to calculate the
            penetrability at

        Returns
        -------
        np.array
            the penetrability values
        
        """
        if self.ell == 0:
            return self.calc_rho(incident_energies)
        

    def calc_cross_section(self, U_matrix, k_sq, inc, out):
        """ Function to calculate the cross section
        for the channel 
        
        Parameters
        ----------
        U_matrix : numpy array
            The U-matrix for the compound nucleus

        k_sq : numpy array
            k values squared for the incident channel
            of the reaction

        inc : int
            The index of the incident channel in the U-matrix

        out : int
            The index of the outgoing channel in the U-matrix

        Returns
        -------
        np.array
            the cross section
        
        """
        self.cross_section = 10**24 * np.pi/k_sq * (1- 2*U_matrix[:,inc,out].real + np.conjugate(U_matrix[:,inc,out])*U_matrix[:,inc,out])

        # check that the imaginary component is basically zero before dropping it
        max_ind = np.argmax(self.cross_section.imag / self.cross_section.real)
        assert self.cross_section[max_ind].imag / self.cross_section[max_ind].real < 1e-10

        self.cross_section = self.cross_section.real