# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit.transpiler import CouplingMap

"""
Utility functions for topological MQA.
"""

class TopoUtil():
    def __init__():
        None
        
    def makeCouple(Lx, Ly, faqe=2, backend="ibm"):
        """Make the customised coupling map for topological MQA. On the each side (left and right),
        there live two fake qubits. We call them as 'faqe'. Those will not only be connected to each other,
        but to qubits that locate on each left and right end of a square lattice.

        Input:
            - Lx (int): How many qubits on the width?
            - Ly (int): How many qubits on the height?
            - faqe (default: 2): The number of 'fake' qubits. They are originally not the part of square lattice.
            - backend (string): the style guide of a square lattice

        Process:
            This coupling map is extended from the square lattice, where there are two 'fake qubits' are
            introduced and both of them are connected to each other. They are not floating around, but
            they are connected to each left and right side of the lattice, making them some kind of a
            torus shape.

        Output:
            - coop (qiskit.transpiler.CouplingMap)
        """
        if (
            backend == "ibm"
        ):  # Stem from the ibm_miami's square lattice + topological MQA approach
            coop = CouplingMap.from_grid(
                Lx, Ly, bidirectional=True
            )  # makes a Lx x Ly rectangular(square?) lattice like IBM.
            """ This is for the automation that isn't planned yet.
            for i in range(faqe): # based on the number of fake qubits, add them onto the coupling map.
                faqe_indices = Lx * Ly + i
                coop.add_physical_qubit(faqe_indices) # being cautious with the indices of those fake ones.
            """
            # Add the physical qubits as qubit indices first.
            # Calculate each fake qubit's indice.
            first_faqe = Lx * Ly + 0
            second_faqe = first_faqe + 1
            coop.add_physical_qubit(first_faqe)
            coop.add_physical_qubit(second_faqe)

            # After adding physical qubit... connect them as we intended
            # Connect the fake qubits first
            coop.add_edge(first_faqe, second_faqe)
            # And...
            for i in range(Ly):
                # ...Connect the left side of the qubit to the square lattice
                coop.add_edge(
                    first_faqe, 0 + Lx * i
                )  # maximum of the i = Ly - 1 so problem solved.
                # ...Connect the right side of the qubit to the square lattice
                coop.add_edge(second_faqe, Lx - 1 + Lx * i)
        elif backend == "iqm":  # Stem from the IQM's square lattice
            coop = CouplingMap()  # Not added yet!
        coop.make_symmetric()  # Make all edges bi-directional.
        return coop

    def checkCouple(cmap):
        """
        Input:
            - cmap: (qiskit.transpiler.CouplingMap)
            
        Process:
            Just prints the coupling map via PIL.
             
        Output:
            prints stuff
        """
        return cmap.draw()  # draws the manual coupling map for topological MQA.
