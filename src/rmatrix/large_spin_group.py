import numpy as np


class LargeSpinGroup:
    def __init__(self, res_energies, incident_channel, energy_grid, debug=False):
        self.res_energies = np.array(res_energies)
        self.incident_channel = incident_channel

        self.energy_grid = np.array(energy_grid)

        # integer dimension values
        self._Nl = len(self.res_energies)
        self._Nc = 1  # num channels
        self._Ne = len(self.energy_grid)

        # create gamma_matrix
        self.gamma_matrix = incident_channel.reduced_width_amplitudes.reshape(
            (self._Nl, 1)
        )

        # create P and L matrices
        self.P_matrix = incident_channel.calc_penetrability(self.energy_grid).reshape(
            (1, 1, self._Ne)
        )

        # create Omega matrix
        self.Omega_matrix = np.exp(
            -1j * incident_channel.calc_rho(self.energy_grid)
        ).reshape((1, 1, self._Ne))

        # create energy matrix
        res_energy_matrix = np.array([np.diag(self.res_energies)] * self._Ne)
        id = np.ones((self._Ne, self._Nl, self._Nl)) * np.identity(self._Nl)
        neutron_energy_matrix = self.energy_grid.reshape(self._Ne, 1, 1) * id
        energy_matrix = res_energy_matrix - neutron_energy_matrix
        self.energy_matrix = energy_matrix.astype(complex)

    def add_channel(self, channel):
        # add widths to gamma matrix
        self.gamma_matrix = np.hstack(
            (self.gamma_matrix, channel.reduced_width_amplitudes.reshape(self._Nl, 1))
        )

        # empty vector to put around the diags
        empty_vector = np.zeros((1, 1, self._Ne))

        # pad right side of P and Omega matrices
        self.P_matrix = np.hstack(
            (self.P_matrix, np.repeat(empty_vector, self._Nc, axis=0))
        )
        self.Omega_matrix = np.hstack(
            (self.Omega_matrix, np.repeat(empty_vector, self._Nc, axis=0))
        )

        # pad left side of new arrays
        new_pen = channel.calc_penetrability(self.energy_grid).reshape((1, 1, self._Ne))
        new_P = np.hstack((np.repeat(empty_vector, self._Nc, axis=1), new_pen))

        new_phase = np.exp(-1j * channel.calc_rho(self.energy_grid)).reshape(
            (1, 1, self._Ne)
        )
        new_Omega = np.hstack((np.repeat(empty_vector, self._Nc, axis=1), new_phase))

        # add in the new array
        self.P_matrix = np.vstack((self.P_matrix, new_P))
        self.Omega_matrix = np.vstack((self.Omega_matrix, new_P))

        self._Nc += 1

    def calc_cross_section(self):
        # reshape the P matrix and Omega matrix
        self.P_matrix = np.moveaxis(self.P_matrix, -1, 0)
        self.Omega_matrix = np.moveaxis(self.Omega_matrix, -1, 0)

        # create P^1/2 and  L-matrix
        self.P_half = np.sqrt(self.P_matrix)

        self.L_matrix = 1j * self.P_matrix
        self.L_inv = np.linalg.inv(self.L_matrix)

        # create gLg matrix
        self.gLg = self.gamma_matrix @ self.L_matrix @ self.gamma_matrix.T

        # create A and A inv
        self.A_inv = self.energy_matrix - self.gLg
        self.A_matrix = np.linalg.inv(self.A_inv)

        id = np.ones((self._Ne, self._Nc, self._Nc)) * np.identity(self._Nc)

        # W
        W_matrix = (
            id
            + 2j
            * self.P_half
            @ self.gamma_matrix.T
            @ self.A_matrix
            @ self.gamma_matrix
            @ self.P_half
        )

        # U
        self.U_matrix = self.Omega_matrix @ W_matrix @ self.Omega_matrix

        # k-sq for in the incident channel
        k_sq = (self.incident_channel.calc_k(self.energy_grid)) ** 2

        # get total cross section for the spin group
        self.total_cross_section = (
            10**24 * 2 * np.pi / k_sq * (1 - self.U_matrix[:, 0, 0].real)
        )
