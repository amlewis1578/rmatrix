import numpy as np 

class SpinGroup:
    def __init__(self, res_energies, incident_channel, outgoing_channels, energy_grid, debug=False):
        self.res_energies = np.array(res_energies)

        if type(outgoing_channels) != list:
            outgoing_channels = [outgoing_channels]

        self.channels = [incident_channel] + outgoing_channels

        self.energy_grid = np.array(energy_grid)

        self.Nl = len(self.res_energies)
        self.Nc = len(self.channels)
        self.Ne = len(self.energy_grid)

        self.set_up_gamma_matrix(debug)
        self.set_up_L_matrix(debug)
        self.set_up_A_matrix(debug)
        self.calc_cross_section(debug)

    def set_up_gamma_matrix(self, debug):
        self.gamma_matrix = np.zeros((self.Nl, self.Nc))
        for i, channel in enumerate(self.channels):
            self.gamma_matrix[:,i] = channel.reduced_width_aplitudes
        if debug: print("\n\ngamma: ", self.gamma_matrix.shape, "\n", self.gamma_matrix)

    def set_up_L_matrix(self, debug):
        self.P_matrix = np.zeros((self.Ne,  self.Nc, self.Nc))
        for i, channel in enumerate(self.channels):
            self.P_matrix[:,i,i] = channel.calc_penetrability(self.energy_grid)
        self.P_half = np.sqrt(self.P_matrix)
        if debug: print("\n\nP: ", self.P_matrix.shape, "\n", self.P_matrix[0])

        self.L_matrix = 1j * self.P_matrix
        self.L_inv = np.linalg.inv(self.L_matrix)

    def set_up_A_matrix(self, debug):
        res_energy_matrix = np.array([np.diag(self.res_energies)]*self.Ne)
        id = np.ones((self.Ne, self.Nl, self.Nl)) * np.identity(self.Nl)
        neutron_energy_matrix = self.energy_grid.reshape(self.Ne,1,1) * id

        energy_matrix = res_energy_matrix - neutron_energy_matrix
        energy_matrix = energy_matrix.astype(complex)

        self.A_inv = energy_matrix - self.gamma_matrix@self.L_matrix@self.gamma_matrix.T
        self.A_matrix = np.linalg.inv(self.A_inv)
        if debug: print("\n\nA: ", self.A_matrix.shape, "\n", self.A_matrix[0])

    def calc_cross_section(self, debug):
        id = np.ones((self.Ne, self.Nc, self.Nc)) * np.identity(self.Nc)

        # W
        W_matrix = id + 2j * self.P_half@self.gamma_matrix.T@self.A_matrix@self.gamma_matrix@self.P_half
        if debug: print("\n\nW: ", W_matrix.shape, "\n", W_matrix[0])

        # U
        Omega_matrix = np.zeros((self.Ne,self.Nc,self.Nc)).astype(complex)
        for i, channel in enumerate(self.channels):
            Omega_matrix[:,i,i] = np.exp(-1j*channel.calc_rho(self.energy_grid))
        # Omega_matrix[:,1,1] = np.exp(-1j*self.capture.calc_rho(self.energy_grid))
        self.U_matrix = Omega_matrix@W_matrix@Omega_matrix

        k_sq = (self.channels[0].calc_k(self.energy_grid))**2
        self.total_cross_section = 10**24 * 2 * np.pi / k_sq * (1 - self.U_matrix[:,0,0].real)

        for i, channel in enumerate(self.channels):
            channel.calc_cross_section(self.U_matrix,k_sq, 0, i )
        
