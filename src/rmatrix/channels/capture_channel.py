from rmatrix.channels.abstract_channel import AbstractChannel
import numpy as np 

class CaptureChannel(AbstractChannel):
    def __init__(self,primary,product,J,pi,ell,ac, excitation, 
                 reduced_width_amplitudes=None, 
                 partial_widths = None, resonance_energies = None):
        """ Class representing a capture channel

        If reduced_width_amplitudes are not given, both partial_widths
        and resonance_energies must be given so that the reduced width
        amplitudes can be calculated. 

        If both reduced_width_amplitudes and partial_widths are given,
        reduced_width_amplitudes will be used and partial_widths will 
        be ignored.
        
        Parameters
        ----------
        primary : Particle object
            The primary gamma

        product : Particle object
            The product nucleus

        J : float
            The spin of the channel

        pi : int
            The parity of the channel, 1 or -1

        ell : int
            Orbital angular momentum for the channel

        ac : float
            The channel radius in 10^(-12) cm
            
        excitation : float
            the excitation energy of the product nucleus in eV

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
        primary : Particle object
            The primary gamma

        product : Particle object
            The product nucleus

        A : int
            Mass number of the product nucleus

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

        excitation : float
            the excitation energy of the product nucleus in eV

        Sn : float
            the neutron separation energy of the product nucleus

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

        # Because the call signature has changed but the major version of the code hasn't,
        # there are some checks here to retain backward-compatibility
        if type(excitation) == list or type(excitation) == np.ndarray:
            print("**WARNING: the order of the parameters for CaptureChannel has changed.")
            print("      The parameter reduced_width_amplitude is no longer required, so it has been moved ")
            print("      after the required parameter excitation. The call signature is now: \n")
            print("\tCaptureChannel(primary: Particle, product: Particle, J: float, pi: int, ell: int,")
            print("\t               ac: float, excitation: float,")
            print("\t           either:")
            print("\t               (1) reduced_width_amplitudes: list or numpy array   OR ")
            print("\t               (2) partial_widths: list or numpy array  AND ")
            print("\t                   resonance_energies: list or numpy array")
            print("\t               )\n")
            print("    To maintain backwards compatibility, if excitation is a list or numpy array, it is assumed")
            print("    that the old call signature was used and the parameters given are interchanged. Check that")
            print("    these are correct before using this object:\n")

            temp = excitation
            excitation = reduced_width_amplitudes
            reduced_width_amplitudes = temp
            print(f'\treduced_width_amplitudes: {reduced_width_amplitudes}')
            print(f'\texcitation: {excitation} eV')

        self.Sn = product.Sn
        super().__init__(primary,product,J,pi,ell,ac,reduced_width_amplitudes,
                         partial_widths, resonance_energies, excitation)
        

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
        hbar = 6.582119e-16  # eV-s
        c = 2.99792e10  # cm/s
        e_gamma = self.Sn + np.array(incident_energies) - self.excitation
        k_cm = e_gamma / (hbar * c)
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
        return (self.calc_k(incident_energies) * self.ac*10**(-12))**(2*self.ell + 1)

    def calc_cross_section(self,U_matrix,k_sq, inc, out):
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
            the cross section values
        
        """
        self.cross_section = 10**24 * np.pi/k_sq * np.conjugate(U_matrix[:,inc,out])*U_matrix[:,inc,out] 
        
        # check that the imaginary component is basically zero before dropping it
        max_ind = np.argmax(self.cross_section.imag)
        assert self.cross_section[max_ind].imag / self.cross_section[max_ind].real < 1e-10
        self.cross_section = self.cross_section.real