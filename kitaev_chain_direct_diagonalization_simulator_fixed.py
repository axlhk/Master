import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import multiprocessing as mp
import ctypes
import os
import sys
import scipy as sp


"""
This program contains a class used for performing simulations of the Kitaev chain
using direct diagonalization methods of the Hamiltonian, and can simulate time 
evolution of a state. 

A method for a Kitaev chain Hamiltonian with a T-junction has also been written 
but has not been tested, so use with caution. 

Documentation is otherwise specified in each class method. 
"""

class KitaevChain:
    def __init__(self, N, t, mu, delta, junction=False, vertical_sites=0, import_ground_state=False, ground_state_file = None, run_tests = True, cutoff=1e-10):
        self.N = N                     # Integer. Number of sites.
        self.t = t                     # Float array. Hopping parameter.
        self.mu = mu                   # Float array. "Chemical potential"
        self.delta = delta             # Float array. Superconducting order parameter
        self.cutoff = cutoff           # Float. Cutoff for numerical errors.
        self.junction = junction       # Boolean, should the system have a T-junction?
        self.N2 = 2**self.N            # Integer. Dimension of Hilbert space.
        if junction:
            self.vertical_sites = vertical_sites                    # Integer. Number of sites in the vertical part of the T-junction.
            self.horizontal_sites = self.N - self.vertical_sites    # Integer. Number of sites in the horizontal part of the T-junction.
        if import_ground_state:
            # do something to load ground state from file
            pass
        self.run_tests = run_tests

    def find_eigenstates(self, gen_hamiltonian=True, H = None):
        """
        Function that finds the eigenstates of the system. Uses brute force
        diagonalization of Hamiltonian over entire Hilbert space.
        """
        # Get Hamiltonian matrix
        if gen_hamiltonian:
            H = self.generate_hamiltonian_matrix(self.mu, self.t, self.delta)

        # Set up empty (complex) array to contain eigenvectors
        self.eigvecs = np.zeros((self.N2,self.N2),dtype='complex')

        # Diagonalize H
        self.eigvals, self.eigvecs = np.linalg.eig(H)

        # Store ground state separately
        self.ground_state = np.zeros(self.N2,dtype='complex')
        self.ground_state[:] = self.eigvecs[:,np.argmin(self.eigvals)]


    #@jit
    def _int_to_binary_array(self, x):
        """
        Voodoo magic function that uses string manipulation to return the binary
        representation of an integer as an N-dimensional array containing the
        digits of the number.
        """
        return np.array([int(i) for i in str(format(x,'0'+str(self.N)+'b'))])


    def _worker(self, mu, t, delta, N, lower, upper, k, d, N2):
        #print("Worker accessed!")3
        H_temp = np.zeros((upper-lower,N2))
        for i in range(upper-lower):
            for j in range(N2):
                binary_array_left = self._int_to_binary_array(lower + i)
                binary_array_right = self._int_to_binary_array(j)
                H_temp[i,j] = self._add_to_elements(mu, t, delta, binary_array_left, binary_array_right, N)
        d[k] = H_temp

    def generate_hamiltonian_matrix(self, mu, t, delta):
        """
        Generates the Hamiltonian of the system given mu, t, and delta as a
        matrix. Could potentially be optimized by being treated as a sparse matrix.

        mu: Float array. Number operator coefficients.
        t: Float array. Hopping parameters.
        delta: Float array. Superconducting order parameters.
        """



        if self.N>=10:
            H = np.zeros((self.N2,self.N2))

            num_processes = mp.cpu_count()-1  # Use the number of available CPU cores minus 1

            manager = mp.Manager()
            d = manager.dict()

            # Create a pool of worker processes
            pool = mp.Pool(processes=num_processes)

            sections = np.linspace(0, self.N2, num_processes)

            for i in range(len(sections)-1):
                lower = int(sections[i])
                upper = int(sections[i+1])
                pool.apply_async(self._worker, args=(mu.copy(), t.copy(), delta.copy(), self.N, lower, upper, i, d, self.N2))


            pool.close()
            pool.join()

            for i in range(len(sections)-1):
                lower = int(sections[i])
                upper = int(sections[i+1])
                H[lower:upper,:] = d[i]

            return H

        else:
            H = np.zeros((self.N2,self.N2))      # Empty Hamiltonian matrix

            # Iterate over Hilbert space
            for i in range(self.N2):
                for j in range(self.N2):
                    # Convert i and j to binary numbers for _add_to_elements
                    binary_array_left = self._int_to_binary_array(i)
                    binary_array_right = self._int_to_binary_array(j)

                    # Get H_ij
                    H[i,j] = self._add_to_elements(mu, t, delta, binary_array_left, binary_array_right, self.N)
            return H


    def generate_hamiltonian_matrix_junction(self, mu, t, delta):
        """
        Generates the Hamiltonian of the system given mu, t, and delta as a
        matrix. Could potentially be optimized by being treated as a sparse matrix.

        mu: Float array. Number operator coefficients.
        t: Float array. Hopping parameters.
        delta: Float array. Superconducting order parameters.
        """
        H = np.zeros((self.N2,self.N2))      # Empty Hamiltonian matrix

        # Iterate over Hilbert space
        for i in range(self.N2):
            for j in range(self.N2):
                # Convert i and j to binary numbers for _add_to_elements
                binary_array_left = self._int_to_binary_array(i)
                binary_array_right = self._int_to_binary_array(j)

                # Get H_ij
                H[i,j] = self._add_to_elements_junction(mu, t, delta, binary_array_left, \
                                                        binary_array_right, self.N, self.N1, \
                                                        self.N_junc)
        return H



    @staticmethod
    @jit(nopython=True)
    def _add_to_elements(mu, t, delta, binary_array_left, binary_array_right, n):
        """
        Function that gets the element H_ij = <i|H|j> where i and j are binary
        numbers (read from right to left above) corresponding to
        binary_array_left (i) and binary_array_right (j). The binary representation
        corresponds to the physical modes of the system and is therefore convenient.
        This method is made static in order to speed up with numba's jit as it
        takes a lot of time to use otherwise. For a brute force calculation this
        step needs to be done millions of times, and it is therefore necessary to
        use some form of speedup technique.

        mu: Float array. Contains the coefficients for the number operator terms
        of the Hamiltonian.
        t: Float array. Hopping parameters between neighbouring sites.
        delta: Float array. Superconducting order parameter.
        binary_array_left: Integer array. Contains binary representation of the number i.
        binary_array_right: Integer array. Contains binary representation of the number j.
        n: Integer. Number of physical sites.
        """

        # Hamiltonian element
        val = 0

        # Boolean array telling where i and j are equal in binary representation
        binary_array_equal = binary_array_left == binary_array_right

        # Iterate over physical sites
        for k in range(n):
            # If i and j are equal in binary representation, add corresponding
            # particle or hole number operator term
            if np.all(binary_array_equal):
                # Particle number operator
                if binary_array_left[-k-1] == 1:
                    val += -mu[k]/2
                # Hole number operator
                if binary_array_left[-k-1] == 0:
                    val += mu[k]/2

            # Superconducting and hopping terms are interactions between different sites,
            # so there are one less of these than the number operator terms.
            if k!=(n-1):
                # If binary_array_equal sums up to n-2 then two digits in the
                # binary representation of i and j are different, and we need to
                # add hopping or superconducting terms.
                if np.sum(binary_array_equal) == n-2:
                    # Hopping to the left
                    if binary_array_left[-k-1] == 1 and binary_array_left[-(k+1)-1] == 0:
                        if binary_array_right[-k-1] == 0 and binary_array_right[-(k+1)-1] == 1:
                             val += -t[k]
                    # Hopping to the right
                    if binary_array_left[-k-1] == 0 and binary_array_left[-(k+1)-1] == 1:
                        if binary_array_right[-k-1] == 1 and binary_array_right[-(k+1)-1] == 0:
                            val += -t[k]
                    # Pairwise creation of particles (delta)
                    if binary_array_left[-k-1] == 1 and binary_array_left[-(k+1)-1] == 1:
                        if binary_array_right[-k-1] == 0 and binary_array_right[-(k+1)-1] == 0:
                            val += delta[k]
                    # Pairwise annihilation of particles (delta)
                    if binary_array_left[-k-1] == 0 and binary_array_left[-(k+1)-1] == 0:
                        if binary_array_right[-k-1] == 1 and binary_array_right[-(k+1)-1] == 1:
                            val += delta[k]
        return val



    @staticmethod
    @jit(nopython=True)
    def _add_to_elements_junction(mu, t, delta, binary_array_left, binary_array_right, n, n1, n_junc):
        """
        Function that gets the element H_ij = <i|H|j> where i and j are binary
        numbers (read from right to left above) corresponding to
        binary_array_left (i) and binary_array_right (j). The binary representation
        corresponds to the physical modes of the system and is therefore convenient.
        This method is made static in order to speed up with numba's jit as it
        takes a lot of time to use otherwise. For a brute force calculation this
        step needs to be done millions of times, and it is therefore necessary to
        use some form of speedup technique.

        mu: Float array. Contains the coefficients for the number operator terms
        of the Hamiltonian.
        t: Float array. Hopping parameters between neighbouring sites.
        delta: Float array. Superconducting order parameter.
        binary_array_left: Integer array. Contains binary representation of the number i.
        binary_array_right: Integer array. Contains binary representation of the number j.
        n: Integer. Number of physical sites.
        """

        # Hamiltonian element
        val = 0

        # Boolean array telling where i and j are equal in binary representation
        binary_array_equal = binary_array_left == binary_array_right

        # Iterate over physical sites
        for k in range(n1):
            # If i and j are equal in binary representation, add corresponding
            # particle or hole number operator term
            if np.all(binary_array_equal):
                # Particle number operator
                if binary_array_left[-k-1] == 1:
                    val += -mu[k]/2
                # Hole number operator
                if binary_array_left[-k-1] == 0:
                    val += mu[k]/2

            # Superconducting and hopping terms are interactions between different sites,
            # so there are one less of these than the number operator terms.
            if k!=(n1-1):
                # If binary_array_equal sums up to n-2 then two digits in the
                # binary representation of i and j are different, and we need to
                # add hopping or superconducting terms.
                if np.sum(binary_array_equal) == n-2:
                    # Hopping to the left
                    if binary_array_left[-k-1] == 1 and binary_array_left[-(k+1)-1] == 0:
                        if binary_array_right[-k-1] == 0 and binary_array_right[-(k+1)-1] == 1:
                             val += -t[k]
                    # Hopping to the right
                    if binary_array_left[-k-1] == 0 and binary_array_left[-(k+1)-1] == 1:
                        if binary_array_right[-k-1] == 1 and binary_array_right[-(k+1)-1] == 0:
                            val += -t[k]
                    # Pairwise creation of particles (delta)
                    if binary_array_left[-k-1] == 1 and binary_array_left[-(k+1)-1] == 1:
                        if binary_array_right[-k-1] == 0 and binary_array_right[-(k+1)-1] == 0:
                            val += delta[k]
                    # Pairwise annihilation of particles (delta)
                    if binary_array_left[-k-1] == 0 and binary_array_left[-(k+1)-1] == 0:
                        if binary_array_right[-k-1] == 1 and binary_array_right[-(k+1)-1] == 1:
                            val += delta[k]

        for k in range(n1, n):
            # If i and j are equal in binary representation, add corresponding
            # particle or hole number operator term
            if np.all(binary_array_equal):
                # Particle number operator
                if binary_array_left[-k-1] == 1:
                    val += -mu[k]/2
                # Hole number operator
                if binary_array_left[-k-1] == 0:
                    val += mu[k]/2

            # Superconducting and hopping terms are interactions between different sites,
            # so there are one less of these than the number operator terms.
            if k!=(n-1):
                # If binary_array_equal sums up to n-2 then two digits in the
                # binary representation of i and j are different, and we need to
                # add hopping or superconducting terms.
                if np.sum(binary_array_equal) == n-2:
                    # Hopping to the left
                    if binary_array_left[-k-1] == 1 and binary_array_left[-(k+1)-1] == 0:
                        if binary_array_right[-k-1] == 0 and binary_array_right[-(k+1)-1] == 1:
                             val += -t[k]
                    # Hopping to the right
                    if binary_array_left[-k-1] == 0 and binary_array_left[-(k+1)-1] == 1:
                        if binary_array_right[-k-1] == 1 and binary_array_right[-(k+1)-1] == 0:
                            val += -t[k]
                    # Pairwise creation of particles (delta)
                    if binary_array_left[-k-1] == 1 and binary_array_left[-(k+1)-1] == 1:
                        if binary_array_right[-k-1] == 0 and binary_array_right[-(k+1)-1] == 0:
                            val += delta[k]
                    # Pairwise annihilation of particles (delta)
                    if binary_array_left[-k-1] == 0 and binary_array_left[-(k+1)-1] == 0:
                        if binary_array_right[-k-1] == 1 and binary_array_right[-(k+1)-1] == 1:
                            val += delta[k]

        if np.sum(binary_array_equal) == n-2:
            if binary_array_left[-n_junc-1] == 1 and binary_array_left[-(n1+1)-1] == 0:
                if binary_array_right[-n_junc-1] == 0 and binary_array_right[-(n1+1)-1] == 1:
                    val += -t[-1]
            if binary_array_left[-n_junc-1] == 0 and binary_array_left[-(n1+1)-1] == 1:
                if binary_array_right[-n_junc-1] == 1 and binary_array_right[-(n1+1)-1] == 0:
                    val += -t[-1]
            if binary_array_left[-n_junc-1] == 1 and binary_array_left[-(n1+1)-1] == 1:
                if binary_array_right[-n_junc-1] == 0 and binary_array_right[-(n1+1)-1] == 0:
                    val += delta[-1]
            if binary_array_left[-n_junc-1] == 0 and binary_array_left[-(n1+1)-1] == 0:
                if binary_array_right[-n_junc-1] == 1 and binary_array_right[-(n1+1)-1] == 1:
                    val += delta[-1]

        return val



    def gradual_disorder_eigenvalues(self, element, max_strength, n_steps, n_eigvals=None, plot_flag=True, save=False):
        """
        Calculates energy levels with a local disorder modeled as a linearly increasing mu and plots the
        results if flag is set to true.

        element: The entry of mu to be increased.
        max_strength: The maximum value of mu[element] is increased by.
        n_steps: The number of values of disorder to take.
        n_eigvals: The number of eigenvalues to plot. If unspecified, all eigenvalues are plotted (not recommended).
        plot_flag: Boolean. If true, the results are plotted.
        """

        # Set n_eigvals if unspecified
        if n_eigvals==None:
            n_eigvals = 2**self.N

        # Initialize array to contain eigenvalues
        self.eigvals_array = np.zeros((n_steps+1,self.N2))

        # Iterate over disorder strengths
        for i in tqdm(range(n_steps+1)):
            # Find eigenstates and values with strength set to current value
            self.find_eigenstates()

            # Store n_eigvals lowest eigenvalues
            self.eigvals_array[i,:] = np.sort(self.eigvals-self.eigvals[np.argmin(self.eigvals)])

            # Increase disorder strength
            self.mu[element] += max_strength/n_steps

        # Plot results
        if plot_flag:
            self._plot_gradual_disorder_eigenvalues(element, max_strength, n_steps, n_eigvals)

        # Save results
        if save:
            name = "local_eigvals/N={}_mu0={}_delta0={}_t0={}_nsteps={}_neigvals={}_element={}_maxstrength={}".format(self.N, self.mu[0], self.delta[0], self.t[0], n_steps, n_eigvals, element, max_strength)
            np.savez(path + name, self.eigvals_array, element, n_steps, n_eigvals, max_strength)


    def _plot_gradual_disorder_eigenvalues(self, element, max_strength, n_steps, n_eigvals=None):
        """
        Companion function to gradual_disorder_eigenvalues. Plots the results of the calculation.
        """

        # Set n_eigvals if unspecified
        if n_eigvals == None:
            n_eigvals = 2**self.N

        # Set x-axis to disorder strength
        x_array = np.arange(0,max_strength,max_strength/(n_steps+1))

        # Plot results
        plt.figure()
        plt.plot(x_array, self.eigvals_array[:,:n_eigvals])
        plt.xlabel(f"Local disorder strength at site {element}")
        plt.ylabel("Eigenvalues")
        plt.title(f"The {n_eigvals} lowest eigenvalues as a function of local disorder strength")
        plt.show()


    """
    # Inner product function, not used as inner product should be standard inner product.
    def _inner_product(self, state1, state2):
        N2 = 2**self.N
        val = 0
        for i in range(N2):
            for j in range(N2):
                binary_array_left = self._int_to_binary_array(i)
                binary_array_right = self._int_to_binary_array(j)
                binary_array_equal = binary_array_left == binary_array_right
                for k in range(self.N):
    """


    @staticmethod
    def _binary_array_to_int(binary_array):
        """
        Converts an array of binary digits to the corresponding integer number.
        """
        val = 0
        for i in range(len(binary_array)):
            val += binary_array[i]*2**i
        return int(val)


    def _operator_action_on_state(self, operator, state, region = None):
        """
        Applies operator to state and returns result.
        Operator: Vector containing operator as list of coefficients for linear combination of original fermionic operators of the system.
        State: Vector containing state as list of coefficients for physical site state basis.
        Region: Tuple containing the region of the state that the operator acts on. If None, the operator acts on the entire state.
        """

        # Set region to entire chain if not specified
        if region==None:
            region = (0,self.N)

        result = np.zeros(self.N2)              # Vector to contain resulting state
        # Iterate over physical fermionic operators
        for k in range(region[0],region[1]):
            # Iterate over Hilbert space
            for i in range(self.N2):
                # Get binary representation of state
                binary_array = self._int_to_binary_array(i)

                # If entry k of binary number is 1 then act with component of annihilation operator k
                if binary_array[-k-1] == 1:
                    # Store result in component of state with entry k set to zero
                    new_binary_array = binary_array.copy()
                    new_binary_array[-k-1] = 0
                    result[self._binary_array_to_int(new_binary_array)] += operator[2*k]*state[i]

                # If entry k of binary number is 0 then act with component of creation operator k
                if binary_array[-k-1] == 0:
                    # Store result in component of state with entry k set to one
                    new_binary_array = binary_array.copy()
                    new_binary_array[-k-1] = 1
                    result[self._binary_array_to_int(new_binary_array)] += operator[2*k+1]*state[i]

        return result


    def _limited_time_evolution(self, initial_state, t):
        """
        Numerically time evolve a state one timestep at a time with the Euler method.
        This method needs cleaning up as there are many pieces that should not be
        adjusted or used, which completely break the simulation.

        initial_state: Vector containing initial state as list of coefficients for physical
                       site state basis.
        t: Float. Timestep lenght.
        """

        # Get relevant Hamiltonian elements
        self.H = self.generate_hamiltonian_matrix(self.mu, self.t, self.delta)
        self.H += self.energy_constant*np.eye(self.N2)

        derivative_components = np.zeros((self.N2),dtype='complex')
        final_state = initial_state.copy()

        for i in range(self.N2):
            #derivative_components[i] = np.sum(complex(0,-1)*self.H[i,:]*initial_state)
            for j in range(self.N2):
                derivative_components[i] += -complex(0,1)*self.H[i,j]*initial_state[j]

        # Numerically integrate over n_steps with current Hamiltonian (Euler method)
        final_state += derivative_components*t
        final_state *= 1/np.linalg.norm(final_state)

        #print(np.max(derivative_components))

        #if hasattr(self, 'H_prev'):
        #    print(np.all(self.H_prev == self.H))

        #self.H_prev = self.H
        #print(sp.linalg.issymmetric(self.H))

        #if hasattr(self, 'prev_state'):
        #    print(np.all(self.prev_state == final_state))

        #self.prev_state = final_state

        return final_state


    def _limited_time_evolution_euler_cromer(self, initial_state, t):
        """
        Numerically time evolve a state one timestep at a time with the Euler method.
        This method needs cleaning up as there are many pieces that should not be
        adjusted or used, which completely break the simulation.

        initial_state: Vector containing initial state as list of coefficients for physical
                       site state basis.
        t: Float. Timestep lenght.
        """

        # Get relevant Hamiltonian elements
        self.H = self.generate_hamiltonian_matrix(self.mu, self.t, self.delta)
        self.H += self.energy_constant*np.eye(self.N2)

        # Get real and imaginary component vectors
        initial_imag = np.imag(initial_state)
        initial_real = np.real(initial_state)

        # Create copies to store result
        final_real = initial_real.copy()
        final_imag = initial_imag.copy()

        # Update real compoenents first
        for i in range(self.N2):
            real_der = np.sum(self.H[i,:]*initial_imag)
            final_real[i] += real_der*t

        # Use updated real components to update imaginary components
        for i in range(self.N2):
            imag_der = -np.sum(self.H[i,:]*final_real)
            final_imag[i] += imag_der*t

        # Numerically integrate over n_steps with current Hamiltonian (Euler method)
        final_state = final_real + complex(0,1)*final_imag
        final_state *= 1/np.linalg.norm(final_state)

        return final_state


    def time_evolution(self,  n_timesteps, t, n_eigvals=None, n_timestepsteps=1, disorder_type='local', \
                       local_site=None, local_max_strength=10, steps_to_reach_max=None, cut_site=None, \
                       steps_to_cut=None, plot_flag=True, on_and_off=False, save=False, save_path=None, \
                       initial_state='ground_state', add_energy_constant=True, integrator='EulerCromer'):
        """
        This function simulates time evolution of the ground state of the system with disorder as specified,
        and then plots and/or saves the results as specified. The time evolution is done by numerically
        integrating the Schrödinger equation with the Euler method, and this specifically happens in the
        _limited_time_evolution() class method specifically. This can be changed, but memory concerns
        should at least be addressed as the Hamiltonian matrix is quite large, and other methods would
        likely need the Hamiltonian for multiple timesteps at once.

        n_timesteps: Integer. Number of timesteps to simulate.
        t: Float. Total simulation time.
        n_eigvals: Integer. Number of eigenvalues to plot. If unspecified, all eigenvalues are plotted
                   (not recommended) and/or saved (recommended).
        n_timestepsteps: Integer. Number of steps to take per timestep. If unspecified, only one step is taken.
                         This is currently deprecated and should be ignored as it has no functional meaning.
        disorder_type: String. Specifies the type of disorder to simulate. Currently supported are 'local' and 'cut'.
        local_site: Integer. Specifies the site to apply local disorder to. If unspecified, the site is set to the
                    middle site of the chain.
        local_max_strength: Float. Specifies the maximum strength of the local disorder. If unspecified,
                            the strength is set to 10. Should really be specified.
        steps_to_reach_max: Integer. Specifies the number of timesteps to reach the maximum strength of the
                            local disorder (energy barrier). If unspecified, the number of steps is set to
                            half the total number of timesteps.
        cut_site: Integer. Specifies the site to cut the chain at. If unspecified, the site is set to the
                  middle connection of the chain.
        steps_to_cut: Integer. Specifies the number of timesteps to fully cut the chain. If unspecified, the
                      number of steps is set to half the total number of timesteps.
        plot_flag: Boolean. If true, the results are plotted.
        on_and_off: Boolean. If true, the disorder is turned off after it has been turned on, with the same
                    number of steps used to turn on (steps_to_reach_max or steps_to_cut).
        save: Boolean. If true, the results are saved to an automatically named file.
        save_path: String. Specifies the path to save the results to. If unspecified, the results are saved
                   at the current place in the directory in a folder named cut_time_evolution or
                   local_time_evolution depending on disorder_type. This folder probably needs to be
                   pregenerated.
        initial_state: String. Specifies the initial state of the system. Currently supported are 'ground_state',
                          'random_state', and 'random_eigenstate'.
        add_energy_constant: Boolean. If true, an energy constant is added to the Hamiltonian to shift the
                             ground state energy to zero. This can reduce numerical error in the simulation.
        """

        # Set n_eigvals if unspecified
        if n_eigvals==None:
            n_eigvals = self.N2

        # Find eigenstates
        self.find_eigenstates()

        # Sort eigvals and eigvecs
        self.eigvecs_original = self.eigvecs[:,np.argsort(self.eigvals)]
        self.eigvals_original = np.sort(self.eigvals)

        # Add energy constant used to shift the Hamiltonian energy so that
        # the ground state has zero energy. This decreases the numerical error
        # on lower energy states under Euler method integration.
        if add_energy_constant:
            self.energy_constant = -self.eigvals_original[0]
            self.eigvals_original += self.energy_constant
        else:
            self.energy_constant = 0

        # Get the parity of the original states
        self.parity_array_eigvecs = self.detect_parity(self.eigvecs_original)

        # Initialize arrays to contain transfer probabilities
        transfer_components_original = np.zeros((n_timesteps, n_eigvals))
        transfer_components_instantaneous = np.zeros((n_timesteps, n_eigvals))

        # Specify initial state
        if initial_state=='ground_state':
            curr_state = self.ground_state
        elif initial_state=='random_state':
            curr_state = np.sqrt(np.random.uniform(0,1,self.N2) * np.exp(1.j * np.random.uniform(0, 2*np.pi, self.N2)))
            curr_state *= 1/np.linalg.norm(curr_state)
            #curr_state = ((self.eigvecs_original[:,0] + self.eigvecs_original[:,1] + self.eigvecs_original[:,2] + self.eigvecs_original[:,3])/2).astype('complex')
        elif initial_state=='random_eigenstate':
            curr_state = self.eigvecs_original[:,np.random.randint(0,self.N2-1)].astype('complex')
        else:
            sys.exit("BAD INPUT: Unrecognized initial state configuration. Please use either 'ground_state' or 'random_state.'")


        # Initialize instantaneous eigenvalue array
        instantaneous_eigvals = np.zeros((n_timesteps, n_eigvals))

        # Iterate over number of timesteps
        for i in tqdm(range(n_timesteps)):

            """Generate disorder/cuts"""
            # Local energy barrier
            if disorder_type=='local':
                # Specify site if unspecified
                if local_site is None:
                    local_site=int(self.N/2)

                # Specify steps to reach max energy barrier if unspecified
                if steps_to_reach_max==None:
                    steps_to_reach_max=int(n_timesteps/2)

                # Add energy barrier
                if i<steps_to_reach_max:
                    self.mu[local_site] += local_max_strength/steps_to_reach_max

                # Turn off energy barrier if specified
                if on_and_off:
                    if i >= steps_to_reach_max and i < 2*steps_to_reach_max:
                        self.mu[local_site] -= local_max_strength/steps_to_reach_max
            # Cut chain by turning off components
            if disorder_type=='cut':
                # Store original parameter values
                if i == 0:
                    orig_val_t = self.t[cut_site]
                    orig_val_delta = self.delta[cut_site]

                # Specify elements to cut at if unspecified
                if cut_site is None:
                    cut_site=int(self.N/2)

                # Specify steps to fully cut if unspecified
                if steps_to_cut==None:
                    steps_to_cut=int(n_timesteps/2)

                # Cut chain
                if i<steps_to_cut:
                    self.t[cut_site] -= orig_val_t/steps_to_cut
                    self.delta[cut_site] -= orig_val_delta/steps_to_cut

                # Merge chain again if specified
                if on_and_off:
                    if i >= steps_to_cut and i < 2*steps_to_cut:
                        self.t[cut_site] += orig_val_t/steps_to_cut
                        self.delta[cut_site] += orig_val_delta/steps_to_cut

            if disorder_type==None:
                pass

            # Time evolve state one timestep
            if integrator=='Euler':
                curr_state = self._limited_time_evolution(curr_state, t/n_timesteps)
            elif integrator=='EulerCromer':
                curr_state = self._limited_time_evolution_euler_cromer(curr_state, t/n_timesteps)
            else:
                sys.exit("Invalid integrator.")

            # Find eigenstates and eigenvalues of instantaneous Hamiltonian
            self.find_eigenstates(gen_hamiltonian=False, H=self.H)

            # Store sorted instantaneous eigenvectors and eigenvalues
            self.eigvecs_instantaneous = self.eigvecs[:,np.argsort(self.eigvals)]
            instantaneous_eigvals[i,:] = np.sort(self.eigvals)

            # Calculate transition probabilities between time-evolved state and both instantaneous and original eigenstates
            for j in range(n_eigvals):
                transfer_components_original[i,j] = np.abs(np.vdot(self.eigvecs_original[:,j],curr_state))**2
                transfer_components_instantaneous[i,j] = np.abs(np.vdot(self.eigvecs_instantaneous[:,j],curr_state))**2

        # Convert parity array to string for labelling purposes
        parity_string_list = []
        for i in range(n_eigvals):
            if self.parity_array_eigvecs[i]==1:
                parity_string_list.append("Even")
            if self.parity_array_eigvecs[i]==-1:
                parity_string_list.append("Odd")
            if self.parity_array_eigvecs[i]==0:
                parity_string_list.append("Mixed")

        # Sort results according to relevance for plotting (not strictly necessary anymore)
        # Also checks if transition probablities are non-zero between eigenstates of different parity
        transfer_components_plot = []
        index_list = []
        transfer_components_instantaneous_plot = []
        check = False
        # Iterate over number of eigenvalues specified
        for i in range(n_eigvals):
            # Get eigenstates with non-zero transition probabilities
            if transfer_components_original[int(n_timesteps/2),i]>self.cutoff:
                # Append transition probabilities
                transfer_components_plot.append(transfer_components_original[:,i])
                #transfer_components_instantaneous_plot.append(transfer_components_instantaneous[:,i])

                # Append index
                index_list.append(i)

                # Check if parity for transitioning state is the same
                if self.parity_array_eigvecs[i]!=self.parity_array_eigvecs[0]:
                    check = True

        if check:
            # Print a warning if parity is not preserved (should only be possible for certain kinds of disorder)
            print("Transfer components are non-zero between eigenstates of different parity.")

        # Subtract added energy constant
        self.eigvals_original -= self.energy_constant
        instantaneous_eigvals -= self.energy_constant

        # Plot results
        if plot_flag:

            # Get array of time
            times = np.arange(0,t,t/n_timesteps)

            # Old plotting syntax, stored for reference
            """
            plt.plot(times, np.asarray(transfer_components_plot).transpose(), label=[f"Eigenvector {i}. " + parity_string_list[i] + " parity. Energy = " + str(self.eigvals[i]) \
                                                        for i in index_list])
            """

            # Set up colormap of original eigenvalues for colorbar
            norm = mpl.colors.Normalize(vmin=np.min(self.eigvals_original), vmax=np.max(self.eigvals_original))
            cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
            cmap.set_array([])

            # Plot original eigenstate transition probabilities
            fig, ax = plt.subplots()
            for i in range(len(index_list)):
                # Plot transition probabilities as a function of time, colored according to eigenvalue
                ax.plot(times, transfer_components_plot[:][i], color=cmap.to_rgba(self.eigvals_original[index_list[i]]))# label=f"Eigenvector {index_list[i]}. " + parity_string_list[index_list[i]] + " parity. Energy = " + str(self.eigvals[index_list[i]]))

            # Initialize colorbar and set axis labels
            fig.colorbar(cmap, ax=ax, label=r'Energy of eigenstate [$\Delta$]')
            ax.set_xlabel(r"Time [$1/\Delta$]")
            ax.set_ylabel(r"Probability")

            # Set title
            if disorder_type=='local':
                ax.set_title(f"Probability of original eigenstate with disorder at site {local_site+1}")

            if disorder_type=='cut':
                ax.set_title(f"Probability of original eigenstate with chain cut (element(s) {cut_site+1})")


            # Plot instantaneous eigenstate transition probabilities
            fig2, ax2 = plt.subplots(sharex=True,sharey=True)

            # Set up colormap of instantaneous eigenvalues for colorbar
            norm2 = mpl.colors.Normalize(vmin=np.min(instantaneous_eigvals), vmax=np.max(instantaneous_eigvals))

            # Old plotting syntax, stored for reference
            """
            points = np.zeros((n_timesteps*len(transfer_components_instantaneous[0,:]),2))
            for i in range(n_timesteps):
                for j in range(len(transfer_components_instantaneous[0,:])):
                    points[j*n_timesteps+i,0] = times[i]
                    points[j*n_timesteps+i,1] = transfer_components_instantaneous[i,j]

            points = points.reshape(-1,1,2)
            segs = np.concatenate([points[:-1],points[1:]],axis=0)
            """
            """
            points = np.zeros((n_timesteps,len(transfer_components_instantaneous[0,:]),2))
            for i in range(n_timesteps):
                for j in range(len(transfer_components_instantaneous[0,:])):
                    points[i,j,0] = times[i]
                    points[i,j,1] = transfer_components_instantaneous[i,j]

            segs=np.zeros([n_timesteps*len(transfer_components_instantaneous[0,:]),2,2])
            for i in range(n_timesteps-1):
                for j in range(len(transfer_components_instantaneous[0,:])):
                    segs[j*n_timesteps+i,0,:] = points[i,j]
                    segs[j*n_timesteps+i,1,:] = points[i+1,j]
            """
            """
            segs= [np.column_stack((times,transfer_components_instantaneous[:,i])) for i in range(len(transfer_components_instantaneous[0,:]))]
            """

            # Plot instantaneous eigenstate transition probabilities as a function of time, colored according to (current) eigenvalue
            for i in range(len(transfer_components_instantaneous[0,:])):
                # Generate a set of points for every timestep and transition probability individually
                points = np.array([times,transfer_components_instantaneous[:,i]]).transpose().reshape(-1,1,2)

                # Generate a set of line segments for every timestep and transition probability individually
                # This contains pairs of points in which to draw lines in between
                segs = np.concatenate([points[:-1], points[1:]], axis=1)

                # Set up a line collection of the elements in segs (useful for plotting many lines)
                lc = mpl.collections.LineCollection(segs,cmap='viridis',norm=norm2)

                # Set the color of the lines according to the instantaneous eigenvalue
                lc.set_array(instantaneous_eigvals[:-1,i])

                # Specify line properties
                lc.set_linewidth(2)

                # Add the line collection to the plot
                line = ax2.add_collection(lc)

            # Initialize colorbar
            fig2.colorbar(line, ax=ax2, label=r'Energy of eigenstate [$\Delta$]')

            # Sepcify axes
            ax2.set_xlim(0-t*0.05,t*1.05)
            ax2.set_ylim(-0.05,1.05)
            ax2.set_xlabel(r"Time [$1/\Delta$]")
            ax2.set_ylabel(r"Probability")

            # Set title
            if disorder_type=='local':
                ax2.set_title(f"Probability of instantaneous eigenstate with disorder at site(s) {local_site+1}")
            if disorder_type=='cut':
                ax2.set_title(f"Probability of instantaneous eigenstate with chain cut (element(s) {cut_site+1})")
            if disorder_type==None:
                ax.set_title("Probabilities of original eigenstates, timestep dt = {}".format(t/n_timesteps))
                ax2.set_title("Probabilities of instantaneous eigenstates, timestep dt = {}".format(t/n_timesteps))
            # Show plots
            plt.show()

        # Save results
        if save:
            # Use as much info in naming as possible so simulations aren't overwritten
            if disorder_type=='local':
                name = "local_time_evolution/N={}_mu0={}_delta0={}_t0={}_ntimesteps={}_neigvals={}_localsite={}_maxstrength={}_t={}_onandoff={}".format(self.N, self.mu[0], self.delta[0], self.t[0], n_timesteps, n_eigvals, local_site, local_max_strength, t, on_and_off)
                np.savez(save_path + name, transfer_components_original, transfer_components_instantaneous, self.eigvals_original, instantaneous_eigvals, self.parity_array_eigvecs, local_site, local_max_strength, n_timesteps, n_eigvals, t, on_and_off)
            if disorder_type=='cut':
                name = "cut_time_evolution/N={}_mu0={}_delta0={}_t0={}_ntimesteps={}_neigvals={}_cutsite={}_t={}_onandoff={}".format(self.N, self.mu[0], self.delta[0], self.t[0], n_timesteps, n_eigvals, cut_site, t, on_and_off)
                np.savez(save_path + name, transfer_components_original, transfer_components_instantaneous, self.eigvals_original, instantaneous_eigvals, self.parity_array_eigvecs, cut_site, n_timesteps, n_eigvals, t, on_and_off)
            if disorder_type is None:
                name = "stationary/N={}_mu0={}_delta0={}_t0={}_ntimesteps={}_neigvals={}_t={}_initialstate={}_{}".format(self.N, self.mu[0], self.delta[0], self.t[0], n_timesteps, n_eigvals, t, initial_state, integrator)
                np.savez(save_path + name, transfer_components_original, self.eigvals_original, self.parity_array_eigvecs, n_timesteps, n_eigvals, t, integrator, initial_state)


    def cut_chain_eigenvalues(self, n_steps, n_eigvals=None, element=None, plot_flag=True, save=False, path=''):
        """
        Function that calculates the eigenvalues of the Hamiltonian while gradually
        cutting the chain by turning off delta and t between two sites. For a true
        cut both need to be switched off. Unless plot_flag is set to False then this
        function also plots the resulting eigenvalues against the percentage that t
        and delta have been reduced by. Element i of the vectors containing t and
        delta corresponds to the interaction between physical sites i+1 and i+2
        (when we start counting from 1).

        n_steps: Integer. Number of steps to take in reducing delta and t.
        n_eigvals: Integer. Number of eigenvalues to plot results for.
        element: Integer. Element of delta and t arrays to be gradually lowered.
        plot_flag: Boolean. Whether or not to plot the results.
        """

        # Set n_eigvals if unspecified
        if n_eigvals==None:
            n_eigvals = self.N2

        # Select element to lower as middle element  if unspecified
        if element==None:
            element = int(self.N/2)-1

        param_values = (self.delta[element],self.t[element])

        # Initialize array to contain eigenvalues
        self.eigvals_array = np.zeros((n_steps+1,n_eigvals))

        # Iterate over delta and t values
        max_imag = 0
        for i in tqdm(range(n_steps+1)):
            # Find eigenstates and values with strength set to current value
            self.find_eigenstates()

            # Store n_eigvals lowest eigenvalues
            self.eigvals_array[i,:] = np.sort(self.eigvals-self.eigvals[np.argmin(self.eigvals)])[:n_eigvals]

            if np.max(np.imag(self.eigvals))>max_imag:
                max_imag = np.max(np.imag(self.eigvals))

            # Decrease delta and t locally
            self.delta[element] -= param_values[0]/n_steps
            self.t[element] -= param_values[1]/n_steps

        print("Maximum imaginary part of eigenvalues discarded: {}".format(max_imag))

        # Plot results
        self._plot_cut_chain_eigenvalues(element, n_steps, n_eigvals)

        # Save results
        if save:
            name = "cut_chain_eigvals/N={}_mu0={}_delta0={}_t0={}_nsteps={}_neigvals={}_element={}".format(self.N, self.mu[0], self.delta[0], self.t[0], n_steps, n_eigvals, element)
            np.savez(path + name, self.eigvals_array, element, n_steps, n_eigvals)


    def _plot_cut_chain_eigenvalues(self, element, n_steps, n_eigvals=None, title=True):
        """
        Companion function to cut_chain_eigenvalues. Plots the results of the
        calculation.
        """

        # Set n_eigvals if unspecified
        if n_eigvals == None:
            n_eigvals = 2**self.N

        # Set x-axis to percentage of hopping and superconducting property turned off
        x_array = np.arange(0,1,1/(n_steps+1))[::-1]

        # Plot results
        plt.figure()
        plt.plot(x_array, self.eigvals_array, color='black')
        plt.gca().invert_xaxis()
        plt.xlabel(r"Percentage of $t$ and $\Delta$ turned off")
        plt.ylabel("Eigenvalues")
        if title:
            plt.title(f"Gradually  cutting chain between sites {element+1} and {element+2}")
        plt.show()

    @staticmethod
    def load_and_plot_cut_chain_eigenvalues(path, file, n_eigvals_plot, title=True, save_for_latex=False, save_path=None):
        """
        Companion function to cut_chain_eigenvalues. Plots the results of the
        calculation.

        path: String. Path to file to be loaded.
        file: String. Name of file to be loaded.
        n_eigvals_plot: Integer. Number of eigenvalues to plot.
        title: Boolean. Whether or not to include a title.
        save_for_latex: Boolean. Whether or not to save the figure in a format
                        suitable for LaTeX.
        save_path: String. Path to save figure to. If unspecified, the figure is
                   saved to the same path as loaded from.

        Note: In order to import into LaTeX the saved svg files should be converted to pdf.
              This can be done with the svg package in LaTeX, but this only works on unix
              systems as of 2.11.2023. If done on a windows system the conversion needs to
              be done manually with Inkscape.
        """

        # Load data
        data = np.load(path + file)
        eigvals_array = data['arr_0']
        element = data['arr_1']
        n_steps = data['arr_2']
        n_eigvals = data['arr_3']

        # Fix n_eigvals plot to not get accidental index errors
        if n_eigvals_plot>n_eigvals:
            n_eigvals_plot = n_eigvals

        # Set path to save figure to the same path as loaded from if unspecified
        if save_path==None:
            save_path = path

        # Set x-axis to percentage of hopping and superconducting property turned off
        x_array = np.arange(0,1,1/(n_steps+1))[::-1]

        # Set up plotting parameters so the formatting fits with LaTeX
        if save_for_latex:
            plt.rcParams["figure.figsize"] = [6/1.2,4/1.2]
            plt.rcParams["font.size"] = 11
            plt.rcParams["figure.autolayout"] = True
            plt.rcParams['font.family']='serif'
            cmfont = font_manager.FontProperties(fname = mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
            plt.rcParams['font.serif'] = cmfont.get_name()
            plt.rcParams['mathtext.fontset'] = 'cm'
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['axes.formatter.use_mathtext'] = True

        # Plot results
        plt.figure()
        plt.plot(x_array, eigvals_array[:,:n_eigvals_plot], color='black')

        # Set up axes
        plt.gca().invert_xaxis()
        plt.xlabel(rf"$\kappa$")
        plt.ylabel("Eigenvalues")

        # Set title
        if title:
            plt.title(f"Gradually  cutting chain between sites {element+1} and {element+2}")

        # Save or show plot depending on flag
        if save_for_latex:
            plt.savefig(save_path + file[:-4] + "n_plot={}".format(n_eigvals_plot) + ".pdf", bbox_inches='tight', pad_inches=0)
        else:
            plt.show()


    def cut_chain_comparison_with_operators(self, element=None, plot_flag=True):
        """
        N should be even

        WIP; ignore for now.
        """

        if element==None:
            element = int(self.N/2)

        self.find_eigenstates()
        self.find_ground_state_operator(lower_bound=0, upper_bound=element, plotting_flag=False)

        initial_states = self.eigvecs
        initial_eigvals = self.eigvals

        self.delta[element] = 0
        self.t[element] = 0

        self.find_eigenstates()

        final_states = self.eigvecs[np.argsort(self.eigvals)]

        ground_state_operator = np.zeros(2*self.N, dtype='complex')
        ground_state_operator[:2*element] = self.fermionic_operators[self.min_idx[0]]

        test_state = self._operator_action_on_state(ground_state_operator,final_states[0])

        inner_prod = np.zeros(self.N2)
        for i in range(self.N2):
            inner_prod[i] = np.abs(np.conjugate(final_states[:,i])@test_state)

        print(inner_prod)


    def detect_parity(self, states):
        """
        Calculate parity of provided states (in physical site state basis),
        and return parities as a list of integers. Even parity is 1, odd parity
        is -1, and mixed parity is 0. States should be columns of provided array.

        states: Array of states in physical site state basis.
        """

        # Initialize array to store results
        parity_array = np.zeros(len(states[0,:]))

        # Set initial value to something other than zero for clarity
        parity_array += 10

        # Iterate over states
        for i in range(len(states[0,:])):
            check1 = True          # Detect even parity
            check2 = True          # Detect odd parity
            for j in range(len(states[:,0])):
                # Find non-zero phsyical site state basis components
                if np.abs(states[j,i])>self.cutoff:
                    binary_array = self._int_to_binary_array(j)
                    if np.sum(binary_array)%2==1:
                        check1 = False            # Set to false if odd parity state has component
                    if np.sum(binary_array)%2==0:
                        check2 = False            # Set to false if even parity state has component

            # Even parity
            if check1 and not check2:
                parity_array[i] = 1

            # Odd parity
            if check2 and not check1:
                parity_array[i] = -1

            # Mixed parity
            if check1 and check2:
                parity_array[i] = 0

        return parity_array

    @staticmethod
    def load_and_plot_cut_time_evolution(path, file, title=True, save_for_latex=False, save_path=None, cutoff_orig = 1e-7, cutoff_inst=1e-7):
        """
        Companion function to time_evolution with disorder_type='cut'. Plots the results of the
        simulation.

        path: String. Path to the folder containing the data file.
        file: String. Name of the data file.
        n_eigvals_plot: Integer. Number of eigenvalues to plot.
        title: Boolean. Whether or not to include a title in the plot.
        save_for_latex: Boolean. Whether or not to save the plot in a format suitable for LaTeX.
        save_path: String. Path to save the plot to. If unspecified, the plot is saved at the current
                   place in the directory.
        cutoff: Float. Cutoff for transition probabilities to be included in the plot.

        Note: In order to import into LaTeX the saved svg files should be converted to pdf.
              This can be done with the svg package in LaTeX, but this only works on unix
              systems as of 2.11.2023. If done on a windows system the conversion needs to
              be done manually with Inkscape.
        """

        # Load data
        data = np.load(path + file)
        transfer_components_original = data['arr_0']
        transfer_components_instantaneous = data['arr_1']
        eigvals_original = data['arr_2']
        instantaneous_eigvals = data['arr_3']
        parity_array_eigvecs = data['arr_4']
        cut_site = data['arr_5']
        n_timesteps = data['arr_6']
        n_eigvals = data['arr_7']
        t = data['arr_8']
        on_and_off = data['arr_9']
        #on_and_off = False

        # Set path to save figure to the same path as loaded from if unspecified
        if save_path==None:
            save_path = path

        # Generate partial name for saving
        save_name = save_path + file[:-4]

        # Set up which components to plot
        #transfer_components_plot = []
        index_list = []
        index_list_inst = []
        #transfer_components_instantaneous_plot = []
        check = False
        for i in range(n_eigvals):
            if np.any(transfer_components_original[:,i]>=cutoff_orig):
                #transfer_components_plot.append(transfer_components_original[:,i])
                index_list.append(i)
                if parity_array_eigvecs[i]!=parity_array_eigvecs[0]:
                    check = True
            if np.any(transfer_components_instantaneous[:,i]>=cutoff_inst):
                #transfer_components_instantaneous_plot.append(transfer_components_instantaneous[:,i])
                index_list_inst.append(i)


        # Array of times
        times = np.arange(0,t,t/n_timesteps)

        """Plot results"""

        # Set up plotting parameters so the formatting fits with LaTeX
        if save_for_latex:
            plt.rcParams["figure.figsize"] = [6/1.5,4/1.5]
            plt.rcParams["font.size"] = 11
            plt.rcParams["figure.autolayout"] = True
            plt.rcParams['font.family']='serif'
            cmfont = font_manager.FontProperties(fname = mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
            plt.rcParams['font.serif'] = cmfont.get_name()
            plt.rcParams['mathtext.fontset'] = 'cm'
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['axes.formatter.use_mathtext'] = True

        # Set up colormap of original eigenvalues for colorbar
        norm = mpl.colors.Normalize(vmin=np.min(eigvals_original), vmax=np.max(eigvals_original))
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
        cmap.set_array([])

        # Plot original eigenstate transition probabilities
        fig, ax = plt.subplots()
        for i in range(len(index_list)):
            # Plot transition probabilities as a function of time, colored according to eigenvalue
            ax.plot(times, transfer_components_original[:,index_list[i]], color=cmap.to_rgba(eigvals_original[index_list[i]]))

        # Initialize colorbar and set axis labels
        fig.colorbar(cmap, ax=ax, label=r'Eigenvalue of eigenstate')
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"Probability")

        # Set title
        if title:
            ax.set_title(f"Probability of original eigenstate with chain cut (element(s) {cut_site+1})")

        if save_for_latex:
            fig.savefig(save_name + "_original.pdf", bbox_inches='tight', pad_inches=0)

        # Plot instantaneous eigenstate transition probabilities
        fig2, ax2 = plt.subplots(sharex=True,sharey=True)

        # Set up colormap of instantaneous eigenvalues for colorbar
        norm2 = mpl.colors.Normalize(vmin=np.min(instantaneous_eigvals), vmax=np.max(instantaneous_eigvals))

        #print(index_list_inst)

        # Plot instantaneous eigenstate transition probabilities as a function of time, colored according to (current) eigenvalue
        for i in range(len(index_list_inst)):
            # Generate a set of points for every timestep and transition probability individually
            points = np.array([times,transfer_components_instantaneous[:,index_list_inst[i]]]).transpose().reshape(-1,1,2)

            # Generate a set of line segments for every timestep and transition probability individually
            segs = np.concatenate([points[:-1], points[1:]], axis=1)

            # Set up a line collection of the elements in segs (useful for plotting many lines)
            lc = mpl.collections.LineCollection(segs,cmap='viridis',norm=norm2)

            # Set the color of the lines according to the instantaneous eigenvalue
            lc.set_array(instantaneous_eigvals[:-1,index_list_inst[i]])

            # Specify line properties
            lc.set_linewidth(2)

            # Add the line collection to the plot
            line = ax2.add_collection(lc)


        # Initialize colorbar
        fig2.colorbar(line, ax=ax2, label=r'Eigenvalue of eigenstate')

        # Sepcify axes
        ax2.set_xlim(0-t*0.05,t*1.05)
        ax2.set_ylim(-0.05,1.05)
        ax2.set_xlabel(r"Time")
        ax2.set_ylabel(r"Probability")

        # Set title
        if title:
            ax2.set_title(f"Probability of instantaneous eigenstate with chain cut (element(s) {cut_site+1})")

        # Save or show plot depending on flag
        if save_for_latex:
            fig2.savefig(save_name + "_instantaneous.pdf", bbox_inches='tight', pad_inches=0)
        else:
            plt.show()




        """
        parity_string_list = []
        for i in range(n_eigvals):
            if self.parity_array_eigvecs[i]==1:
                parity_string_list.append("Even")
            if self.parity_array_eigvecs[i]==-1:
                parity_string_list.append("Odd")
            if self.parity_array_eigvecs[i]==0:
                parity_string_list.append("Mixed")
        """

    @staticmethod
    def load_and_plot_gradual_disorder_eigenvalues(path, file, n_eigvals_plot, title=True, save_for_latex=False, save_path=None):
        """
        Companion function to gradual_disorder_eigenvalues. Plots the results of the
        calculation.

        path: String. Path to file to be loaded.
        file: String. Name of file to be loaded.
        n_eigvals_plot: Integer. Number of eigenvalues to plot.
        title: Boolean. Whether or not to include a title.
        save_for_latex: Boolean. Whether or not to save the figure in a format
                        suitable for LaTeX.
        save_path: String. Path to save figure to. If unspecified, the figure is
                   saved to the same path as loaded from.

        Note: In order to import into LaTeX the saved svg files should be converted to pdf.
              This can be done with the svg package in LaTeX, but this only works on unix
              systems as of 2.11.2023. If done on a windows system the conversion needs to
              be done manually with Inkscape.
        """

        # Load data
        data = np.load(path + file)
        eigvals_array = data['arr_0']
        element = data['arr_1']
        n_steps = data['arr_2']
        n_eigvals = data['arr_3']
        max_strength = data['arr_4']

        # Fix n_eigvals plot to not get accidental index errors
        if n_eigvals_plot>n_eigvals:
            n_eigvals_plot = n_eigvals

        # Set path to save figure to the same path as loaded from if unspecified
        if save_path==None:
            save_path = path

        # Set x-axis to strength of energy barrier
        x_array = np.arange(0,max_strength,max_strength/(n_steps+1))

        # Set up plotting parameters so the formatting fits with LaTeX
        if save_for_latex:
            plt.rcParams["figure.figsize"] = [6/1.8,4/1.8]
            plt.rcParams["font.size"] = 11
            plt.rcParams["figure.autolayout"] = True
            plt.rcParams['font.family']='serif'
            cmfont = font_manager.FontProperties(fname = mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
            plt.rcParams['font.serif'] = cmfont.get_name()
            plt.rcParams['mathtext.fontset'] = 'cm'
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['axes.formatter.use_mathtext'] = True

        # Plot results
        plt.figure()
        plt.plot(x_array, eigvals_array[:,:n_eigvals_plot], color='black')

        # Set up axes
        plt.xlabel(rf"$\epsilon$")
        plt.ylabel("Eigenvalues")

        # Set title
        if title:
            plt.title(f"Gradually increasing energy at site(s) {element+1}")

        # Save or show plot depending on flag
        if save_for_latex:
            plt.savefig(save_path + file[:-4] + "_nplot={}".format(n_eigvals_plot) + ".pdf", bbox_inches='tight', pad_inches=0)
        else:
            plt.show()


    @staticmethod
    def load_and_plot_local_time_evolution(path, file, title=True, save_for_latex=False, save_path=None, \
                                           cutoff_orig = 1e-7, cutoff_inst=1e-7, plot_every = 1):
        """
        Companion function to time_evolution with disorder_type='cut'. Plots the results of the
        simulation.

        path: String. Path to the folder containing the data file.
        file: String. Name of the data file.
        n_eigvals_plot: Integer. Number of eigenvalues to plot.
        title: Boolean. Whether or not to include a title in the plot.
        save_for_latex: Boolean. Whether or not to save the plot in a format suitable for LaTeX.
        save_path: String. Path to save the plot to. If unspecified, the plot is saved at the current
                   place in the directory.
        cutoff: Float. Cutoff for transition probabilities to be included in the plot.

        Note: In order to import into LaTeX the saved svg files should be converted to pdf.
              This can be done with the svg package in LaTeX, but this only works on unix
              systems as of 2.11.2023. If done on a windows system the conversion needs to
              be done manually with Inkscape.
        """

        # Load data
        data = np.load(path + file)
        transfer_components_original = data['arr_0']
        transfer_components_instantaneous = data['arr_1']
        eigvals_original = data['arr_2']
        instantaneous_eigvals = data['arr_3']
        parity_array_eigvecs = data['arr_4']
        cut_site = data['arr_5']
        max_strength = data['arr_6']
        n_timesteps = data['arr_7']
        n_eigvals = data['arr_8']
        t = data['arr_9']
        on_and_off = data['arr_10']

        # Set path to save figure to the same path as loaded from if unspecified
        if save_path==None:
            save_path = path

        # Generate partial name for saving
        save_name = save_path + file[:-4]

        # Set up which components to plot
        #transfer_components_plot = []
        index_list = []
        index_list_inst = []
        #transfer_components_instantaneous_plot = []
        check = False
        for i in range(n_eigvals):
            if np.any(transfer_components_original[:,i]>=cutoff_orig):
                #transfer_components_plot.append(transfer_components_original[:,i])
                index_list.append(i)
                if parity_array_eigvecs[i]!=parity_array_eigvecs[0]:
                    check = True
            if np.any(transfer_components_instantaneous[:,i]>=cutoff_inst):
                #transfer_components_instantaneous_plot.append(transfer_components_instantaneous[:,i])
                index_list_inst.append(i)


        # Array of times
        times = np.arange(0,t,t/n_timesteps)

        """Plot results"""

        # Set up plotting parameters so the formatting fits with LaTeX
        if save_for_latex:
            plt.rcParams["figure.figsize"] = [6/1.7,4/1.6]
            plt.rcParams["font.size"] = 11
            plt.rcParams["figure.autolayout"] = True
            plt.rcParams['font.family']='serif'
            cmfont = font_manager.FontProperties(fname = mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
            plt.rcParams['font.serif'] = cmfont.get_name()
            plt.rcParams['mathtext.fontset'] = 'cm'
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['axes.formatter.use_mathtext'] = True

        # Set up colormap of original eigenvalues for colorbar
        norm = mpl.colors.Normalize(vmin=np.min(eigvals_original), vmax=np.max(eigvals_original))
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
        cmap.set_array([])

        # Plot original eigenstate transition probabilities
        fig, ax = plt.subplots()
        for i in range(len(index_list)):
            # Plot transition probabilities as a function of time, colored according to eigenvalue
            ax.plot(times[::plot_every], transfer_components_original[::plot_every,index_list[i]], color=cmap.to_rgba(eigvals_original[index_list[i]]))

        # Initialize colorbar and set axis labels
        fig.colorbar(cmap, ax=ax, label=rf'Eigenvalue of eigenstate')
        ax.set_xlabel(r"Time")
        ax.set_ylabel(rf"$|c_i|^2$")

        # Set title
        if title:
            ax.set_title(f"Probability of original eigenstate with local disorder (element(s) {cut_site+1})")

        if save_for_latex:
            fig.savefig(save_name + "_original.pdf", bbox_inches='tight', pad_inches=0)

        # Plot instantaneous eigenstate transition probabilities
        fig2, ax2 = plt.subplots(sharex=True,sharey=True)

        # Set up colormap of instantaneous eigenvalues for colorbar
        norm2 = mpl.colors.Normalize(vmin=np.min(instantaneous_eigvals), vmax=np.max(instantaneous_eigvals))

        #print(index_list_inst)

        # Plot instantaneous eigenstate transition probabilities as a function of time, colored according to (current) eigenvalue
        for i in range(len(index_list_inst)):
            # Generate a set of points for every timestep and transition probability individually
            points = np.array([times[::plot_every],transfer_components_instantaneous[::plot_every,index_list_inst[i]]]).transpose().reshape(-1,1,2)

            # Generate a set of line segments for every timestep and transition probability individually
            segs = np.concatenate([points[:-1], points[1:]], axis=1)

            # Set up a line collection of the elements in segs (useful for plotting many lines)
            lc = mpl.collections.LineCollection(segs,cmap='viridis',norm=norm2)

            # Set the color of the lines according to the instantaneous eigenvalue
            lc.set_array(instantaneous_eigvals[:-1,index_list_inst[i]])

            # Specify line properties
            lc.set_linewidth(2)

            # Add the line collection to the plot
            line = ax2.add_collection(lc)


        # Initialize colorbar
        fig2.colorbar(line, ax=ax2, label=rf'Eigenvalue of eigenstate')

        # Sepcify axes
        ax2.set_xlim(0-t*0.05,t*1.05)
        ax2.set_ylim(-0.05,1.05)
        ax2.set_xlabel(r"Time")
        ax2.set_ylabel(rf"$|d_i|^2$")

        # Set title
        if title:
            ax2.set_title(f"Probability of instantaneous eigenstate with local disorder (element(s) {cut_site+1})")

        # Save or show plot depending on flag
        if save_for_latex:
            fig2.savefig(save_name + "_instantaneous.pdf", bbox_inches='tight', pad_inches=0)
        else:
            plt.show()


    @staticmethod
    def load_and_plot_stationary_time_evolution(path, file, title=True, save_for_latex=False, save_path=None, cutoff = 1e-7):
        """
        Companion function to time_evolution with disorder_type=None. Plots the results of the
        simulation.

        path: String. Path to the folder containing the data file.
        file: String. Name of the data file.
        title: Boolean. Whether or not to include a title in the plot.
        save_for_latex: Boolean. Whether or not to save the plot in a format suitable for LaTeX.
        save_path: String. Path to save the plot to. If unspecified, the plot is saved at the current
                   place in the directory.
        cutoff: Float. Cutoff for transition probabilities to be included in the plot.

        Note: In order to import into LaTeX the saved svg files should be converted to pdf.
              This can be done with the svg package in LaTeX, but this only works on unix
              systems as of 2.11.2023. If done on a windows system the conversion needs to
              be done manually with Inkscape.
        """

        # Load data
        data = np.load(path + file)
        transfer_components_original = data['arr_0']
        eigvals_original = data['arr_1']
        parity_array_eigvecs = data['arr_2']
        n_timesteps = data['arr_3']
        n_eigvals = data['arr_4']
        t = data['arr_5']
        integrator = data['arr_6']
        initial_state = data['arr_7']

        # Set path to save figure to the same path as loaded from if unspecified
        if save_path==None:
            save_path = path

        # Generate partial name for saving
        save_name = save_path + file[:-4]

        # Set up which components to plot
        #transfer_components_plot = []
        index_list = []
        #transfer_components_instantaneous_plot = []
        check = False
        for i in range(n_eigvals):
            if np.any(transfer_components_original[:,i]>=cutoff):
                #transfer_components_plot.append(transfer_components_original[:,i])
                index_list.append(i)
                if parity_array_eigvecs[i]!=parity_array_eigvecs[0]:
                    check = True


        # Array of times
        times = np.arange(0,t,t/n_timesteps)

        """Plot results"""

        # Set up plotting parameters so the formatting fits with LaTeX
        if save_for_latex:
            plt.rcParams["figure.figsize"] = [6/1.7,4/1.6]
            plt.rcParams["font.size"] = 11
            plt.rcParams["figure.autolayout"] = True
            plt.rcParams['font.family']='serif'
            cmfont = font_manager.FontProperties(fname = mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
            plt.rcParams['font.serif'] = cmfont.get_name()
            plt.rcParams['mathtext.fontset'] = 'cm'
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['axes.formatter.use_mathtext'] = True

        # Set up colormap of original eigenvalues for colorbar
        norm = mpl.colors.Normalize(vmin=np.min(eigvals_original), vmax=np.max(eigvals_original))
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
        cmap.set_array([])

        # Plot original eigenstate transition probabilities
        fig, ax = plt.subplots()
        for i in range(len(index_list)):
            # Plot transition probabilities as a function of time, colored according to eigenvalue
            ax.plot(times, transfer_components_original[:,index_list[i]], color=cmap.to_rgba(eigvals_original[index_list[i]]))

        # Initialize colorbar and set axis labels
        fig.colorbar(cmap, ax=ax, label=r'Eigenvalue of eigenstate')
        ax.set_xlabel(r"Time")
        ax.set_ylabel(rf"$|c_i|^2$")

        # Set title
        if title:
            ax.set_title(f"Probability of eigenstates with stationary Hamiltonian")

        if save_for_latex:
            fig.savefig(save_name + ".pdf", bbox_inches='tight', pad_inches=0)
        else:
            plt.show()



if __name__=="__main__":
    np.random.seed(seed=1248587)

    N = 10
    mu = np.zeros(N)
    mu[:] = 0.5
    t = np.zeros(N)
    t[:] = 1
    delta = np.zeros(N)
    delta[:] = 2
    unit_factor = 1
    path=''
    figures_path=''

    # Initialize chain
    kitaev_chain = KitaevChain(N, t, mu, delta)

    #t=0, delta=0 chain cutting time evo
    T = 20
    nt = 1000
    kitaev_chain.time_evolution(nt, T, n_eigvals=None, n_timestepsteps=1, disorder_type='cut', cut_site=int(N/2), \
                                steps_to_cut=int(nt/2), on_and_off=True, plot_flag=True, save=True, save_path=path)