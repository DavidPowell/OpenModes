# -*- coding: utf-8 -*-
"""
OpenModes - An eigenmode solver for open electromagnetic resonantors
Copyright (C) 2013 David Powell

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import numpy as np

from utils import SingularSparse
#import core_for

# a constant, indicating that this material is a perfect electric conductor
PecMaterial = "Perfect electric conductor"

class Triangles(object):
    """A class for storing triangles of a surface mesh
    """
    
    def __init__(self, N_tri):
        self.N_tri = N_tri
        self.nodes = np.empty((N_tri, 3), np.int32, order="F")
        self.area = np.empty(N_tri, np.float64, order="F")    # area of triangle
        self.lens = np.empty((N_tri, 3), np.float64, order="F") # lengths of opposite edges
        
    def __len__(self):
        return self.N_tri

#class RwgBasis(object):
#    """A class for storing RWG basis functions
#    
#    The basis functions are coordinate indepedent, so they are not changed
#    by any operations performed on the object (except maybe scale and shear???)
#    """
#    
#    def __init__(self, N_basis):
#        self.N_basis = N_basis
#        self.tri_p = np.empty(N_basis, np.int32) # index of T+
#        self.tri_m = np.empty(N_basis, np.int32) # index of T-
#        self.node_p = np.empty(N_basis, np.int32) # internal index of free node of T+
#        self.node_m = np.empty(N_basis, np.int32) # internal index of free node of T-
#        self.rho_cp = np.empty((N_basis, 3), np.float64, order="F") # centre of T+
#        self.rho_cm = np.empty((N_basis, 3), np.float64, order="F") # centre of T-
#        self.len = np.empty(N_basis, np.float64) # length of shared edge (fairly redundant)
#
#    def __len__(self):
#        return self.N_basis


class LibraryPart(object):
    """A physical part available to be placed within the simulation

    A part should correspond to the smallest unit of interest in the 
    simulation and must not be connected with or overlap any other object

    Note that a `LibraryPart` cannot be modified, as it is designed to be an
    unchanging reference object

    Note that the coordinate origin of the object is important, this will be
    used in certain calculations (e.g. dipole moments), and should correspond
    to the geometric centre of the object.
    """

    # TODO: how does connection to parent change if the objects are in 
    # different positions within a layered background?
    
    # TODO: what if a parent object is moved - this may invalidate
    # child objects in layered media
    
    def __init__(self, nodes, triangles, loop_star = True):
        """
        Parameters
        ----------
        nodes : ndarray
            The nodes making up the object
        triangles : ndarray
            The node indices of triangles making up the object
        loop_star : boolean, optional
            If True, then will contain members `loop_basis` and `star_basis`
            representing a transformation of the impedance matrix to a loop
            and star based representation
        """
        
        #self.singular_terms = None

        self.nodes = nodes
        N_tri = len(triangles)
        N_nodes = len(nodes)

        self.tri = Triangles(N_tri)
        self.tri.nodes = triangles

        # indexing: triangle, vertex_num, x/y/z
        vertices = self.nodes[self.tri.nodes]
    
        # each edge is numbered according to its opposite node
        self.tri.lens = np.sqrt(np.sum((np.roll(vertices, 1, axis=1) - 
                        np.roll(vertices, 2, axis=1))**2, axis=2))    

        self.shortest_edge = self.tri.lens.min()
     
        # find all the shared edges, these correspond to the basis functions
        # also find which triangles share a given edge
        # use of lists is slow but convenient (change to OrderedSet?)
        all_edges = set()
        shared_edges = [] 
        sharing_triangles_dict = dict()
        
        # allow each node to find out which triangles it is part of
        self.triangle_nodes = [set() for _ in xrange(N_nodes)]

        for count, t_nodes in enumerate(self.tri.nodes):    
            #t_nodes = self.tri.nodes[count]
            # calculate the area of each triangle
            vec1 = self.nodes[t_nodes[1]]-self.nodes[t_nodes[0]]
            vec2 = self.nodes[t_nodes[2]]-self.nodes[t_nodes[0]]
            self.tri.area[count] = 0.5*np.sqrt(sum(np.cross(vec1, vec2)**2))
    
            # edges are represented as sets to avoid ambiguity order                
            edges = [frozenset((t_nodes[0], t_nodes[1])), 
                     frozenset((t_nodes[0], t_nodes[2])), 
                     frozenset((t_nodes[1], t_nodes[2]))]
            for edge in edges:
                if edge in all_edges:
                    shared_edges.append(edge)
                    sharing_triangles_dict[edge].append(count)
                else:
                    all_edges.add(edge)
                    sharing_triangles_dict[edge] = [count]
    
            # tell each node that it is a part of this triangle
            for node in t_nodes:
                self.triangle_nodes[node].add(count)            
    
        N_basis = len(shared_edges)

        self.basis = RwgBasis(N_basis)

        # allow each triangle to know its neighbours with which
        # it shares an edge    eta_0 = np.sqrt(mu_0/epsilon_0)

        triangle_edge_neighbours = [set() for _ in xrange(N_tri)]

        for count, edge in enumerate(shared_edges):
            # set the RWG basis function triangles
            tri_p = sharing_triangles_dict[edge][0]
            tri_m = sharing_triangles_dict[edge][1]
            
            self.basis.tri_p[count] = tri_p
            self.basis.tri_m[count] = tri_m

            # tell the triangles that they share an edge, which basis function
            # this is and the relative sign of the RWG basis function
            triangle_edge_neighbours[tri_p].add((tri_m, count, 1))
            triangle_edge_neighbours[tri_m].add((tri_p, count, -1))
            
            # also find the non-shared and shared nodes
            shared_nodes = []
            for count_node in self.tri.nodes[self.basis.tri_p[count]]:
                if count_node in self.tri.nodes[self.basis.tri_m[count]]:
                    shared_nodes.append(count_node)
                else:
                    unshared_p = count_node
                    
            for count_node in self.tri.nodes[self.basis.tri_m[count]]:
                if count_node not in shared_nodes:
                    unshared_m = count_node
    
            # determine the indices of the unshared nodes, indexed within the
            # sharing triangles (i.e. 0, 1 or 2)
            self.basis.node_p[count] = np.where(self.tri.nodes[self.basis.tri_p[count]] == unshared_p)[0][0]
            self.basis.node_m[count] = np.where(self.tri.nodes[self.basis.tri_m[count]] == unshared_m)[0][0]
          
            # find the edge lengths
            self.basis.len[count] = np.sqrt(sum((self.nodes[shared_nodes[0]]-self.nodes[shared_nodes[1]])**2))
            
            # and the triangle centre coordinates in terms of rho
            self.basis.rho_cp[count] = (np.sum(self.nodes[shared_nodes], axis=0)+self.nodes[unshared_p])/3.0-self.nodes[unshared_p]
            self.basis.rho_cm[count] = self.nodes[unshared_m]-(np.sum(self.nodes[shared_nodes], axis=0)+self.nodes[unshared_m])/3.0    

            # make all elements unwriteable ??
            self.nodes.flags.writeable = False


        if loop_star:
            
            # [1] G. Vecchi, “Loop-star decomposition of basis functions in the 
            # discretization of the EFIE,” IEEE Transactions on Antennas and 
            # Propagation, vol. 47, no. 2, pp. 339–346, 1999.
            # [2] J.-F. Lee, R. Lee, and R. J. Burkholder, “Loop star basis 
            # functions and a robust preconditioner for EFIE scattering
            # problems,” IEEE Transactions on Antennas and Propagation, 
            # vol. 51, no. 8, pp. 1855–1863, Aug. 2003.
            
            # now find the loop-star decomposition
            # TODO: loop and star basis should be sparse arrays
            # TODO: loop and star basis should be merged              

            # Now create the RWG to star conversion matrix
            # One star exists for each triangle  
            # CHECK: do loop/star bases need different normalisation??             
            self.star_basis = np.zeros((N_tri, N_basis))
            #self.star_basis = sp.lil_matrix((N_tri, N_basis), dtype=np.float64)
       
            for count in xrange(N_basis):
                # set the RWG basis function triangles
                # set the star basis functions
                self.star_basis[self.basis.tri_p[count], count] = 1.0
                self.star_basis[self.basis.tri_m[count], count] = -1.0

            # find the set of unshared edges
            unshared_edges = all_edges - set(shared_edges)
    
            # then find all the boundary nodes
            outer_nodes = set()
            for edge in unshared_edges:
                outer_nodes.update(edge)
    
            # find the nodes which don't belong to any shared edge
            self.inner_nodes = set(xrange(N_nodes)) - outer_nodes

            # eliminate nodes which don't belong to any edge at all 
            # (e.g. the point at the centre when constructing an arc)
            for node_count in xrange(N_nodes):
                if len(self.triangle_nodes[node_count]) == 0:
                    #print "eliminated node", node_count
                    self.inner_nodes.remove(node_count)

            n_vertices = len(self.inner_nodes)+len(outer_nodes)
            #print "vertices", n_vertices
            #print "faces", len(triangles)
            #print "edges", len(all_edges)
            boundary_contours = 2-n_vertices+len(all_edges)-len(triangles)
            #print "separated contours", boundary_contours

            
            # create the RWG to loop conversion matrix
            # Note that this would create one basis function for each inner 
            # node which may exceed the number of RWG degrees of freedom. In
            # this case arbitrary loops at the end of the list are dropped in
            # the conversion
            
            n_loop = len(self.inner_nodes) + boundary_contours - 1
            
            if n_loop == 0:
                self.loop_basis = np.zeros((0, N_basis), dtype=np.float64)
            else:
            
                self.loop_basis = np.zeros((n_loop, N_basis))        
                #self.loop_basis = sp.lil_matrix((n_loop, N_basis),
                #dtype=np.float64)
               
                #self.inner_nodes = list(self.inner_nodes)
                for inner_node_number, node_number in enumerate(list(self.inner_nodes)[:n_loop]):
                     # all triangles sharing this node
                    loop_triangles = self.triangle_nodes[node_number]
                    ordered_triangles = []
                    ordered_triangles.append(tuple(loop_triangles)[0])
                    
                    # find the next triangle in the loop
                    for next_triangle, shared_edge, edge_weight in triangle_edge_neighbours[ordered_triangles[-1]]:
                        if next_triangle in loop_triangles:
                            ordered_triangles.append(next_triangle)
                            self.loop_basis[inner_node_number, abs(shared_edge)] = edge_weight
                            break
                     
                    # loop through all other triangles
                    while next_triangle != ordered_triangles[0]:
                        for next_triangle, shared_edge, edge_weight in triangle_edge_neighbours[ordered_triangles[-1]]:
                            if next_triangle in loop_triangles and next_triangle != ordered_triangles[-2]:
                                ordered_triangles.append(next_triangle)
                                self.loop_basis[inner_node_number, abs(shared_edge)] = edge_weight
                                break
    
            # normalise both star basis transformation to unit magnitude
            # RWG basis is already normalised to area/2, so this yields a total
            # Gram matrix quite close to unity
            self.star_basis /= np.sqrt(np.sum(abs(self.star_basis), axis=1))[:, None]
            self.loop_basis /= np.sqrt(np.sum(abs(self.loop_basis), axis=1))[:, None]



    def precalc_singular(self, quadrature_rule):
        """Precalculate the singular impedance terms for an object
    
        Parameters
        ----------
        quadrature_rule : tuple of 2 ndarrays
            the barycentric coordinates and weights of the quadrature to
            use for the non-analytical neighbour terms
            
        Returns
        -------
        singular_terms : SingularSparse object
            the sparse array of singular impedance terms
        
        """
        
        # TODO: when a substrate is included, the Green's function should
        # be queried to hash the transform matrix in such a way that all
        # objects with the same hash have the same singular parts of their
        # impedance matrix.
        
        if hasattr(self, "singular_terms"):
            return self.singular_terms                

        xi_eta_eval, weights = quadrature_rule        
        
        # Precalculate the singular integration rules for faces, which depend
        # on the observation point    
        triangles = self.tri
        N_face = len(triangles)
    
        singular_terms = SingularSparse()
        # find the neighbouring triangles (including self terms) to integrate
        # singular part
        for p in xrange(0, N_face): # observer:
            
            nodes_p = self.nodes[triangles.nodes[p]]

            sharing_triangles = set()
            for node in triangles.nodes[p]:
                sharing_triangles = sharing_triangles.union(self.triangle_nodes[node])
            
            # find any neighbouring elements which are touching
            for q in sharing_triangles:
                if q == p:
                    # calculate the self term using the exact formula
                    singular_terms[p, p] = core_for.arcioni_singular(nodes_p,)
                else:
                    # at least one node is shared
                    # calculate neighbour integrals semi-numerically
                    nodes_q = self.nodes[triangles.nodes[q]]
                    singular_terms[p, q] = core_for.face_integrals_hanninen(
                                        nodes_q, xi_eta_eval, weights, nodes_p)
        
        self.singular_terms = singular_terms
        return singular_terms

    @property
    def loop_star_transform(self):
        """The transformation from RWG to loop-star basis functions"""
        return np.vstack((self.loop_basis, self.star_basis[:-1, :]))

    def face_currents_and_charges(self, I, for_integration=False, 
                                  loop_star = True):
        r"""Calculate the charges and currents at the centre of each face
        
        Note that the charge is actually scaled by a factor of -1j*omega
        
        Parameters
        ----------                
        I : ndarray
            current solution vector
        for_integration : boolean, optional
            if True, then the current and charge effectively multiplied by 
            surface area

        Returns
        -------
        rmid : ndarray
            triangle mid-points
        face_currents : ndarray
            current at each mid point
        face_charges : ndarray
            total charge on each triangle, multiplied by $j \omega$
        """
        
        nodes = self.nodes
        tri = self.tri
        basis = self.basis
    
        num_basis = len(basis)
        num_triangles = len(tri)
    
        face_currents = np.zeros((num_triangles, 3), np.complex128)
        face_charges = np.zeros((num_triangles), np.complex128)
        
        if loop_star:
            I = np.dot(self.loop_star_transform.T, I)
        
        # calculate the centre of each triangle        
        rmid = np.sum(nodes[tri.nodes], axis=1)/3.0
    
        for count in xrange(num_basis):
            # positive contribution from edges
            which_face = basis.tri_p[count]
            if for_integration:
                face_currents[which_face] += basis.rho_cp[count]*I[count]/2
                face_charges[which_face] += I[count]
                
            else:
                area = tri.area[which_face]
                face_currents[which_face] += basis.rho_cp[count]*basis.len[count]*I[count]/(2*area)
                face_charges[which_face] += basis.len[count]*I[count]/area
                
        
            # negative contribution from edges
            which_face = basis.tri_m[count]
            if for_integration:
                face_currents[which_face] += basis.rho_cm[count]*I[count]/2
                face_charges[which_face] -= I[count]
                
            else:
                area = tri.area[which_face]
                face_currents[which_face] += basis.rho_cm[count]*basis.len[count]*I[count]/(2*area)
                face_charges[which_face] -= basis.len[count]*I[count]/area
    
        return rmid, face_currents, face_charges
    
class SimulationPart(object):
    """A part which has been placed into the simulation, and which can be
    modified"""

    def __init__(self, mesh, notify_function, material=PecMaterial,
                 location = None):

        self.mesh = mesh
        self.material = material
        self.notify = notify_function

        self.initial_location = location
        self.reset()
        #self.transformation_matrix = np.eye(4)
        #if location is not None:
        #    self.translate(location)

    #def notify(self):
    #    """Notify that this part has been changed"""
    #    self.notify_function()
        
    def reset(self):
        """Reset this part to the default values of the original `LibraryPart`
        from which this `SimulationPart` was created
        """
        
        self.transformation_matrix = np.eye(4)
        if self.initial_location is not None:
            self.translate(self.initial_location)
        else:
            self.notify()

    @property
    def nodes(self):
        "The nodes of this part after all transformations have been applied"
        return np.dot(self.transformation_matrix[:3, :3], 
              self.mesh.nodes.T).T + self.transformation_matrix[:3, 3]
        
    def translate(self, offset_vector):
        """Translate a part by an arbitrary offset vector
        
        Care needs to be take if this puts an object in a different layer
        """
        # does not break relationship with parent
        #self.nodes = self.nodes+np.array(offset_vector)
         
        translation = np.eye(4)
        translation[:3, 3] = offset_vector
         
        self.transformation_matrix = np.dot(translation, self.transformation_matrix)
         
        self.notify() # reset any combined mesh this is a part of
           
    def rotate(self, axis, angle):
        """
        Rotate about an arbitrary axis        
        
        Parameters
        ----------
        axis : ndarray
            the vector about which to rotate
        angle : number
            angle of rotation in degrees
        
        Algorithm taken from
        http://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_parameters
        """

        # TODO: enable rotation about arbitrary coordinates, and about the
        # centre of the object        
        
        axis = np.array(axis)
        axis /= np.sqrt(np.dot(axis, axis))
        
        angle *= np.pi/180.0        
        
        a = np.cos(0.5*angle)
        b, c, d = axis*np.sin(0.5*angle)
        
        matrix = np.array([[a**2 + b**2 - c**2 - d**2, 2*(b*c - a*d), 2*(b*d + a*c), 0],
                           [2*(b*c + a*d), a**2 + c**2 - b**2 - d**2, 2*(c*d - a*b), 0],
                           [2*(b*d - a*c), 2*(c*d + a*b), a**2 + d**2 - b**2 - c**2, 0],
                           [0, 0, 0, 1]])
        
        self.transformation_matrix = np.dot(matrix, self.transformation_matrix)
        self.notify()

    def scale(self, scale_factor):
        raise NotImplementedError
        # non-affine transform, will cause problems

        # TODO: work out how scale factor affects pre-calculated 1/R terms
        # and scale them accordingly (or record them if possible for scaling
        # at some future point)

        # also, not clear what would happen to dipole moment

    def shear(self):
        raise NotImplementedError
        # non-affine transform, will cause MAJOR problems
 
