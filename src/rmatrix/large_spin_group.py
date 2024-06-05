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
        self.L_matrix = 1j * self.P_matrix

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

        # add penetrabilities to the P matrix
        empty_vector = np.zeros(
            (1, 1, self._Ne)
        )  # empty vector to put around the diags
        # pad right side of P matrix with zeros
        self.P_matrix = np.hstack(
            (self.P_matrix, np.repeat(empty_vector, self._Nc, axis=0))
        )

        # pad left side of new array
        new_pen = channel.calc_penetrability(self.energy_grid).reshape((1, 1, self._Ne))
        new = np.hstack((np.repeat(empty_vector, self._Nc, axis=1), new_pen))

        # add in the new array
        self.P_matrix = np.vstack((self.P_matrix, new))

        self._Nc += 1
