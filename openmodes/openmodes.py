# -*- coding: utf-8 -*-
"""
OpenModes
---------

A Method of Moments (Boundary Element Method) code designed to find the modes
of open resonators such as meta-atoms, (nano) antennas, scattering particles
etc.

Using these modes, broadband models of these elements can be created, enabling
excitation, coupling between them and scattering to be solved easily, and
broadband models to be created

Copyright 2013 David Powell

TODO: License to go here
"""
from __future__ import division, print_function

import os.path as osp

# numpy and scipy
import numpy as np
import scipy.linalg as la
from utils import fortran_real_type, SingularSparse

from scipy.constants import epsilon_0, mu_0, c
     
f_real = fortran_real_type#()

import integration

# compiled fortran libraries
import core_for

# Multiprocessing pool cannot be directly applied to fortran routines,
# therefore wrap them in pure python functions
#def arcioni_singular(*args):
#    return core_for.arcioni_singular(*args)
#
#def face_integrals_hanninen(*args):
#    return core_for.face_integrals_hanninen(*args)

class Triangles(object):
    """A class for storing triangles of a surface mesh
    
    Should be largely compatible with a recordarray, but avoids array
    copy operations when passing to fortran subroutines
    """
    
    def __init__(self, N_tri):
        self.N_tri = N_tri
        self.nodes = np.empty((N_tri, 3), np.int32, order="F")
        self.area = np.empty(N_tri, f_real, order="F")    # area of triangle
        self.lens = np.empty((N_tri, 3), f_real, order="F") # lengths of opposite edges
        
    def __len__(self):
        return self.N_tri

class RwgBasis(object):
    """A class for storing RWG basis functions
    
    The basis functions are coordinate indepedent, so they are not changed
    by any operations performed on the object (except maybe scale and shear???)
    """
    
    def __init__(self, N_basis):
        self.N_basis = N_basis
        self.tri_p = np.empty(N_basis, np.int32) # index of T+
        self.tri_m = np.empty(N_basis, np.int32) # index of T-
        self.node_p = np.empty(N_basis, np.int32) # internal index of free node of T+
        self.node_m = np.empty(N_basis, np.int32) # internal index of free node of T-
        self.rho_cp = np.empty((N_basis, 3), f_real, order="F") # centre of T+
        self.rho_cm = np.empty((N_basis, 3), f_real, order="F") # centre of T-
        self.len = np.empty(N_basis, f_real) # length of shared edge (fairly redundant)

    def __len__(self):
        return self.N_basis


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
                
        
        #if self.parent_precalc_link:
        #    return self.parent.precalc_singular(quadrature_rule)
        
        xi_eta_eval, weights = quadrature_rule        
        
        # TODO: parallelism should not just be within each object ??
        #pool = multiprocessing.Pool()    

        # this object holds all the asynchronous results
        #result_objects = {}        
    
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
                    #result_objects[(p, p)] = pool.apply_async(arcioni_singular, (nodes_p,))
                    singular_terms[p, p] = core_for.arcioni_singular(nodes_p,)
                else:
                    # at least one node is shared
                    # calculate neighbour integrals semi-numerically
                    nodes_q = self.nodes[triangles.nodes[q]]
                    #result_objects[(p,q)] = pool.apply_async(face_integrals_hanninen, (nodes_q, xi_eta_eval, weights, nodes_p))
                    singular_terms[p, q] = core_for.face_integrals_hanninen(
                                        nodes_q, xi_eta_eval, weights, nodes_p)
        
        #pool.close()
        # all the results have been asynchronously started
        # now fetch them sequentially and add to the sparse matrix
        #for (p, q), result in result_objects.iteritems():
        #    singular_terms[p, q] = result.get()
         
        #pool.join()            

        self.singular_terms = singular_terms
        #self.self_precalc_valid = True

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
        
    def get_dipole_moments(self, I, omega, loop_star = True, electric_order = 1, 
                           magnetic_order=1):
        """Calculate the electric and magnetic multipole moments up to the
        specified order
        
        Parameters
        ----------
        I : ndarray
            the current vector of the relevant mode or excitation

        Returns
        -------
        p : ndarray
            electric dipole moment
        m : ndarray
            magnetic dipole moment
            
        Moments are calculated relative to zero coordinate - does not affect
        the electric dipole, but will affect the magnetic dipole moment and
        any other higher-order multipoles
        
        The moments are 'primitive moments' as defined by Raab and de Lange
        """
        rmid, face_currents, face_charges = self.face_currents_and_charges(I, 
                                    for_integration=True, loop_star=loop_star)
        
        electric_moments = []
        magnetic_moments = []
        
        # electric dipole moment
        if electric_order >= 1:
            electric_moments.append(np.sum(rmid[:, :]*face_charges[:, None], 
                                           axis=0)/(1j*omega))
        
        # electric quadrupole moment
        if electric_order >= 2:        
            quad = np.empty((3, 3), np.complex128)
            for i in xrange(3):
                for j in xrange(3):
                    quad[i, j] = np.sum(rmid[:, i]*rmid[:, j]*face_charges[:])
            electric_moments.append(quad/(1j*omega))
                    
        if magnetic_order >= 1:
            magnetic_moments.append(0.5*np.sum(np.cross(rmid[:, :], 
                                                face_currents[:, :]), axis=0))
        
        return electric_moments, magnetic_moments

def load_parts(filename, mesh_tol=None, force_tuple = False):
    """
    Open a gmsh geometry or mesh file into the relevant parts
    
    Parameters
    ----------
    filename : string
        The name of the file to open. Can be a gmsh .msh file, or a geometry
        file, which will be meshed first
    mesh_tol : float, optional
        If opening a geometry file, it will be meshed with this tolerance
    force_tuple : boolean, optional
        Ensure that a tuple is always returned, even if only a single part
        is found in the file

    Returns
    -------
    parts : tuple
        A tuple of `SimulationParts`, one for each separate geometric entity
        found in the gmsh file
    """
    
    import gmsh    
    
    if osp.splitext(osp.basename(filename))[1] == ".msh":
        # assume that this is a binary mesh already generate by gmsh
        meshed_name = filename
    else:
        # assume that this is a gmsh geometry file, so mesh it first
        meshed_name = gmsh.mesh_geometry(filename, mesh_tol)

    node_tri_pairs = gmsh.read_mesh(meshed_name)
    
    parts = tuple(LibraryPart(nodes, triangles) for (nodes, triangles) in node_tri_pairs)
    if len(parts) == 1 and not force_tuple:
        return parts[0]
    else:
        return parts


class SimulationPart(object):
    """A part which has been placed into the simulation, and which can be
    modified"""

    def __init__(self, library_part, notify_function):

        self.library_part = library_part
        self.notify_function = notify_function
        self.reset()

    def notify(self):
        """Notify that this part has been changed"""
        self.notify_function()
        
    def precalc_singular(self, quadrature_rule):
        #TODO: if layered geometry is used, precalculation cannot simply rely
        # on the parent object
        return self.library_part.precalc_singular(quadrature_rule)

    def reset(self):
        """Reset this part to the default values of the original `LibraryPart`
        from which this `SimulationPart` was created
        """
        
        self.tri = self.library_part.tri
        self.basis = self.library_part.basis

        self.transformation_matrix = np.eye(4)
        self.shortest_edge = self.library_part.shortest_edge

        # copy the loop and star basis conversion if the parent has them
        if (hasattr(self.library_part, "loop_basis") and 
            hasattr(self.library_part, "star_basis")):
            self.loop_basis = self.library_part.loop_basis
            self.star_basis = self.library_part.star_basis
            
        self.notify()

    @property
    def nodes(self):
        "The nodes of this part after all transformations have been applied"
        return np.dot(self.transformation_matrix[:3, :3], self.library_part.nodes.T).T + self.transformation_matrix[:3, 3]
        
    def get_dipole_moments(self, I, omega, loop_star = True, electric_order=1,
                           magnetic_order=1):
        """Calculate the dipole moments
        
        They are transformed versions of the parent's assuming that only
        affine transformation have been performed
        """
        
        # transformations currently disabled        
        
        return self.library_part.get_dipole_moments(I, omega, loop_star, 
                                                electric_order, magnetic_order) 
        
        #p, m = self.parent.get_dipole_moments(I, omega)
        #return np.dot(self.transformation_matrix[:3, :3], p), np.dot(self.transformation_matrix[:3, :3], m)

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
 
 
class Simulation(object):
    """A class representing the method of moments solver and data structures
    
    The object can be pickled (e.g. for parallel simulations). However, the
    unpickled copies will break the change watching machinery. This is not
    a problem if the geometry is not changed in the parallel engines (which
    would make no sense anyway)
    """

    def __init__(self, integration_rule = 5):
        """       
        Parameters
        ----------
        integration_rule : integer
            the order of the integration rule on triangles
        """
        
        self.quadrature_rule = integration.get_dunavant_rule(integration_rule)
        
        self.triangle_quadrature = {}
        self.singular_integrals = {}
               
        self._parts = []

    def place_part(self, library_part):
        """Add a part to the simulation domain
        
        Parameters
        ----------
        library_part : LibraryPart
            The part to place
            
        Returns
        -------
        sim_part : SimulationPart
            The part placed in the simulation
            
        The part will be placed at the origin. It can be translated, rotated
        etc using the relevant methods of `SimulationPart`            
            
        Currently the part can only be modelled as a perfect electric
        conductor
        """
        
        sim_part = SimulationPart(library_part, self.parts_modified)
        self._parts.append(sim_part)

        #sim_part.add_observer(self.objects_changed)
        #self.objects_changed()
        self.parts_modified()
        return sim_part
        
    def parts_modified(self):
        """Called when any parts have been modified, invalidating the combined 
        mesh and precalculated impedance terms"""
        
        if hasattr(self, "combined_mesh"):
            del self._combined_mesh
            
        if hasattr(self, "combined_precalc"):
            del self._combined_precalc
        
  
    def object_basis(self, loop_star = True):
        """Find the ranges of the impedance matrix and source vectors 
        corresponding to each MOM_object which has been added
        
        Parameters
        ----------
        loop_star : bool
            Whether or not to calculate in loop-star basis
        
        Returns
        -------
        slices : list of `slice`
            each slice object gives the relevant part of the matrix/vector
        """
        
        if loop_star:
            return ([slice(a, b) for a, b in self.combined_mesh['loop_ranges']],
                    [slice(a, b) for a, b in self.combined_mesh['star_ranges']])
                    
        else:
            return [slice(a, b) for a, b in self.combined_mesh['objs'].basis]


    def object_triangles(self):
        """Find the ranges of triangle faces
        corresponding to each MOM_object which has been added
        
        Returns
        -------
        slices : list of `slice`
            each slice object gives the relevant part of the matrix/vector
            for each object
        """
        
        sizes = [0]+[len(obj.tri) for obj in self._parts]
        sizes = np.cumsum(sizes)
        return [slice(sizes[i], sizes[i+1]) for i in xrange(len(sizes)-1)]
        
    @property
    def combined_mesh(self):
        """Combine the meshes of all objects into a global mesh

        If the combined mesh is invalid then recombine it
        
        Returns
        -------
        nodes : ndarray
        triangles : Triangles
        basis : RwgBasis
        objs
        max_distance
        shortest_edge : number
            the shortest of all inter-node distances
        
        """
        try:
            return self._combined_mesh
        except AttributeError:
            N_objs = len(self._parts)
            objs = np.rec.recarray(N_objs, dtype=[("nodes", np.int32, 2), 
                                                  ("triangles", np.int32, 2), 
                                                  ("basis", np.int32, 2)])
        
            # calculate the total number of nodes and triangles    
            N_nodes = sum(len(obj.nodes) for obj in self._parts)
            N_tri = sum(len(obj.tri) for obj in self._parts)
            N_basis = sum(len(obj.basis) for obj in self._parts)
        
            nodes = np.empty((N_nodes, 3), f_real, order='F')
            tri = Triangles(N_tri)                                  
            basis = RwgBasis(N_basis)
        
            # merge all the nodes, triangles and basis functions into a
            # master array
            offset_nodes = 0
            offset_tri = 0
            offset_basis = 0
            
            shortest_edge = np.inf
        
            loop_basis = []
            star_basis = []
        
            for obj_count, obj in enumerate(self._parts):
        
                nodes[offset_nodes:offset_nodes+len(obj.nodes)] = obj.nodes
        
                # renumber triangle nodes to global indices
                tri_range = slice(offset_tri, offset_tri+len(obj.tri))
                tri.nodes[tri_range] = obj.tri.nodes+offset_nodes
                tri.area[tri_range] = obj.tri.area
                tri.lens[tri_range] = obj.tri.lens
                #tri.mid[tri_range] = obj.tri.mid
        
                # renumber the basis functions
                basis_range = slice(offset_basis, offset_basis+len(obj.basis))
                basis.tri_p[basis_range] = obj.basis.tri_p+offset_tri
                basis.tri_m[basis_range] = obj.basis.tri_m+offset_tri
                basis.node_p[basis_range] = obj.basis.node_p
                basis.node_m[basis_range] = obj.basis.node_m
        
                basis.rho_cp[basis_range] = obj.basis.rho_cp
                basis.rho_cm[basis_range] = obj.basis.rho_cm
                basis.len[basis_range] = obj.basis.len
        
                # record the ranges of each array corresponding to each object            
                objs[obj_count].nodes[0] = offset_nodes
                objs[obj_count].nodes[1] = offset_nodes+len(obj.nodes)
                objs[obj_count].triangles[0] = offset_tri
                objs[obj_count].triangles[1] = offset_tri+len(obj.tri)
                objs[obj_count].basis[0] = offset_basis
                objs[obj_count].basis[1] = offset_basis+len(obj.basis)
        
                offset_nodes += len(obj.nodes)
                offset_tri += len(obj.tri)
                offset_basis += len(obj.basis)
        
                shortest_edge = min(shortest_edge, obj.shortest_edge)
                
                # get loop and star basis information from children
                try:
                    loop_basis.append(obj.loop_basis)
                    # arbitrarily chose the last star of each element to drop
                    star_basis.append(obj.star_basis[:-1, :])
                except AttributeError:
                    pass

            # find the largest distance between nodes within the structure
            max_distance = np.sqrt(np.sum((nodes[:, None, :]-nodes[None, :, :])**2, axis=2)).max()

            # Store the parts of the combined mesh which don't depend on the loop
            # star basis functions being defined
            self._combined_mesh = {'nodes' : nodes, 'tri' : tri, 
                                   'basis' : basis, 'objs' : objs, 
                                   'max_distance' : max_distance,
                                   'shortest_edge' : shortest_edge}

            # only keep loop and star basis if all children have it
            if len(loop_basis) != N_objs or len(star_basis) != N_objs:
                #loop_basis = None
                #star_basis = None
                pass
            else:
                n_loop = np.hstack(([0], 
                                np.cumsum([l.shape[0] for l in loop_basis])))
                n_J = np.hstack(([0], 
                                 np.cumsum([l.shape[1] for l in loop_basis])))
                
                # scipy sparse matrices cannot have zero dimensions        
                if n_loop[-1] == 0:
                    loop_basis_comb = np.zeros((0, N_basis))
                else:
                    loop_basis_comb = np.zeros((n_loop[-1], n_J[-1]), np.float32)
                    #loop_basis_comb = sp.lil_matrix((n_loop[-1], N_basis), dtype=np.float64)
                    for index, sub_matrix in enumerate(loop_basis):
                        loop_basis_comb[n_loop[index]:n_loop[index+1], 
                                        n_J[index]:n_J[index+1]] = sub_matrix
    
                    # loop_basis_comb = loop_basis_comb.tocsr()

                n_star = np.hstack(([0], np.cumsum([s.shape[0] for s in star_basis])))
                star_basis_comb = np.zeros((n_star[-1], n_J[-1]), np.float32)
                #star_basis_comb = sp.lil_matrix((n_star[-1], N_basis), dtype=np.float64)
                for index, sub_matrix in enumerate(star_basis):
                    star_basis_comb[n_star[index]:n_star[index+1], 
                                    n_J[index]:n_J[index+1]] = sub_matrix
                
                #star_basis_comb = star_basis_comb.tocsr()

                # scipy sparse matrices cannot have zero dimensions        
                if n_loop[-1] == 0:
                    loop_star_transform = star_basis_comb
                else:
                    #loop_star_transform = sp.vstack((loop_basis, star_basis)).tocsr()
                    loop_star_transform = np.vstack((loop_basis_comb, star_basis_comb))
                
                self._combined_mesh['loop_star_transform'] = loop_star_transform 
                self._combined_mesh['n_loop'] = n_loop[-1]

                # the ranges into the impedance matrix for the loops and stars of
                # each object
                self._combined_mesh['loop_ranges'] = np.array([[n_loop[i], n_loop[i+1]] for i in xrange(N_objs)])
                self._combined_mesh['star_ranges'] = np.array([[n_star[i], n_star[i+1]] for i in xrange(N_objs)])+n_loop[-1]
            
            return self._combined_mesh

    @property
    def combined_precalc(self):
        """Combine the precalculated data of several parts
        
        If the combined precalc data is invalid then recalculate it"""
        
        try:
            return self._combined_precalc    
        except AttributeError:
            
            offset_tri = 0
            precalc_sparse = SingularSparse()
        
            for part in self._parts:
        
                singular_terms = part.precalc_singular(self.quadrature_rule)
        
                # the pre-calculated self-impedance data
                for key, value in singular_terms.iteritems():
                    precalc_sparse[(key[0]+offset_tri, key[1]+offset_tri)] = value
                
                offset_tri += len(part.tri)
        
            self._combined_precalc =  precalc_sparse.tocsr()
            return self._combined_precalc


    # simplify access to the nodes and triangles using properties    
    @property
    def nodes(self):
        """The combined nodes of all objects in this simulation"""
        return self.combined_mesh['nodes']

    @property
    def tri(self):
        """The combined triangles of all objects in this simulation"""
        return self.combined_mesh['tri']

    @property
    def basis(self):
        """The combined basis functions of all objects in this simulation"""
        return self.combined_mesh['basis']
        
    @property
    def loop_star_transform(self):
        """The transformation from RWG to loop-star basis functions"""
        return self.combined_mesh['loop_star_transform']

    @property
    def n_loop(self):
        """The number of loop basis functions"""
        return self.combined_mesh['n_loop']

    @property
    def triangle_to_rwg(self):
        """The sparse matrix which converts from triangle to RWG basis"""
        return self.combined_mesh['triangle_to_rwg']

  
    def impedance_matrix(self, s, serial_interpolation = False, 
                         loop_star = True):
        """Evaluate the impedances matrices
        
        Parameters
        ----------        
        s : number
            complex frequency at which to calculate impedance (in rad/s)
        serial_interpolation : boolean, optional
            do interpolation of Green's function serially
            which is slower, but easier to debug
        loop_star : boolean, optional
            transform impedance from RWG to loop-star basis
        
        This version assumes that the mesh is associated with each object,
        which is then used to build a single global mesh        
        
        Returns
        -------
        L, S : ndarray
            the inductance and susceptance matrices
         
        """
        
        xi_eta_eval, weights = self.quadrature_rule

        nodes = self.nodes
        basis = self.basis
        tri = self.tri

        (I_A_sing, I_phi_sing, index_sing, indptr_sing) = self.combined_precalc
   
        A_faces, phi_faces = core_for.z_efie_faces(nodes, tri.nodes, s, 
           xi_eta_eval, weights, I_phi_sing, I_A_sing, index_sing, indptr_sing)

#        A_faces, phi_faces = Z_EFIE_faces(nodes, tri.nodes, s, 
#           xi_eta_eval, weights, I_phi_sing, I_A_sing, index_sing, indptr_sing)

        
        #import core_cython
        L, S = core_for.triangle_face_to_rwg(basis.tri_p, basis.tri_m, 
                                basis.node_p, basis.node_m, A_faces, phi_faces)
        
        L *= mu_0/(4*np.pi)
        S *= 1/(np.pi*epsilon_0)

                                             
        if loop_star:
            # perform transformation to loop-star basis
            
            L = self.loop_star_transform.dot(L.dot(self.loop_star_transform.T))
            S = self.loop_star_transform.dot(S.dot(self.loop_star_transform.T))
            
            n_loop = self.n_loop
            S[:n_loop, :] = 0.0
            S[:, :n_loop] = 0.0            

        return L, S

    def sub_matrix(self, matrix, part1, part2 = None, loop_star = True):
        """Return a sub-matrix, corresponding to the self term for a particular
        object or mutual term between a pair of objects
        
        Parameters
        ----------
        matrix: ndarray
            the impedance matrix, or one of its components
        part1: number
            which part to calculate matrix for
        part2: number, optional
            Second part. If not specified, self matrix given by part1 will
            be returned
        loop_star: boolean, optional
            Whether loop-star basis functions are used
            
        Returns
        -------
        sub_matrix : ndarray
            The requested subsection of the matrix
        """

        if part2 is None:
            part2 = part1
        
        if loop_star:
            loop_ranges, star_ranges = self.object_basis()
            
            loop1 = loop_ranges[part1]
            loop2 = loop_ranges[part2]

            star1 = star_ranges[part1]
            star2 = star_ranges[part2]

            return np.vstack((np.hstack((matrix[loop1, loop2], 
                                         matrix[loop1, star2])), 
                              np.hstack((matrix[star1, loop2], 
                                         matrix[star1, star2])) ))
            
        else:
            raise NotImplementedError
        
    def source_term(self, e_inc, jk_inc, loop_star = True):
        """Evaluate the source vector due to the incident wave
        
        Parameters
        ----------        
        e_inc: ndarray
            incident field polarisation in free space
        jk_inc: ndarray
            incident wave vector in free space
            
        Returns
        -------
        V : ndarray
            the source "voltage" vector
        """

        xi_eta_eval, weights = self.quadrature_rule
        
        nodes = self.nodes
        tri = self.tri
        basis = self.basis

        incident = core_for.voltage_plane_wave(nodes, tri.nodes, basis.tri_p, 
                                       basis.tri_m, basis.node_p, basis.node_m, 
                                       xi_eta_eval, weights, e_inc, jk_inc)

        if loop_star:
            incident = self.loop_star_transform.dot(incident)

        return incident

    def linearised_eig(self, L, S, n_modes, which_obj = None):
        """Solves a linearised approximation to the eigenvalue problem from
        the impedance calculated at some fixed frequency.
        
        Parameters
        ----------
        L, S : ndarray
            The two components of the impedance matrix. They *must* be
            calculated in the loop-star basis.
        n_modes : int
            The number of modes required.
        which_obj : int, optional
            Which object in the system to find modes for. If not specified, 
            then modes of the entire system will be found
            
        Returns
        -------
        omega_mode : ndarray, complex
            The resonant frequencies of the modes
        j_mode : ndarray, complex
            Columns of this matrix contain the corresponding modal currents
        """
        
        if which_obj is None:
            # just solve for the whole system, which is easy
            
            loop_range = slice(0, self.n_loop)
            star_range = slice(self.n_loop, len(self.basis))
            
        else:
            loop_ranges, star_ranges = self.object_basis()
            loop_range = loop_ranges[which_obj]
            star_range = star_ranges[which_obj]

        if loop_range.start == loop_range.stop:
            # object has no loops
            no_loops = True
        else:
            no_loops = False
            
        if no_loops:
            L_red = L[star_range, star_range]
        else:
            L_conv = la.solve(L[loop_range, loop_range], 
                              L[loop_range, star_range])
            L_red = L[star_range, star_range] - np.dot(L[star_range, loop_range], L_conv)

        # find eigenvalues, and star part of eigenvectors, for LS combined modes
        w, v_s = la.eig(S[star_range, star_range], L_red)
        
        if no_loops:
            vr = v_s
        else:
            v_l = -np.dot(L_conv, v_s)
            vr = np.vstack((v_l, v_s))
        
        w_freq = np.sqrt(w)/2/np.pi
        w_selected = np.ma.masked_array(w_freq, w_freq.real < w_freq.imag)
        which_modes = np.argsort(w_selected.real)[:n_modes]
        
        return np.sqrt(w[which_modes]), vr[:, which_modes]

def Z_EFIE_faces(nodes, triangle_nodes, s, xi_eta_eval, weights, phi_precalc, 
                 A_precalc, indices_precalc, indptr_precalc):
    """
    ! Calculate the face to face interaction terms used to build the impedance matrix
    !
    ! As per Rao, Wilton, Glisson, IEEE Trans AP-30, 409 (1982)
    ! Uses impedance extraction techqnique of Hanninen, precalculated
    !
    ! nodes - position of all the triangle nodes
    ! basis_tri_p/m - the positive and negative triangles for each basis function
    ! basis_node_p/m - the free nodes for each basis function
    ! omega - evaulation frequency in rad/s
    ! s - complex frequency
    ! xi_eta_eval, weights - quadrature rule over the triangle (weights normalised to 0.5)
    ! A_precalc, phi_precalc - precalculated 1/R singular terms

    use core_for
    implicit none

    integer, intent(in) :: num_nodes, num_triangles, num_integration, num_singular
    ! f2py intent(hide) :: num_nodes, num_triangles, num_integration, num_singular

    real(WP), intent(in), dimension(0:num_nodes-1, 0:2) :: nodes
    integer, intent(in), dimension(0:num_triangles-1, 0:2) :: triangle_nodes

    complex(WP), intent(in) :: s

    real(WP), intent(in), dimension(0:num_integration-1, 0:1) :: xi_eta_eval
    real(WP), intent(in), dimension(0:num_integration-1) :: weights

    real(WP), intent(in), dimension(0:num_singular-1) :: phi_precalc
    real(WP), intent(in), dimension(0:num_singular-1, 3, 3) :: A_precalc
    integer, intent(in), dimension(0:num_singular-1) :: indices_precalc
    integer, intent(in), dimension(0:num_triangles) :: indptr_precalc
    """

#    complex(WP), intent(out), dimension(0:num_triangles-1, 0:num_triangles-1, 0:2, 0:2) :: A_face
#    complex(WP), intent(out), dimension(0:num_triangles-1, 0:num_triangles-1) :: phi_face
#    
#    
#    complex(WP) :: jk_0 
#    
#    real(WP), dimension(0:2, 0:2) :: nodes_p, nodes_q
#    complex(WP) :: A_part, phi_part
#    complex(WP), dimension(3, 3) :: I_A
#    complex(WP) :: I_phi
#
#    integer :: p, q, q_p, q_m, p_p, p_m, ip_p, ip_m, iq_p, iq_m, m, n, index_singular

    num_triangles = triangle_nodes.shape[0]
    num_integration = weights.shape[0]

    jk_0 = s/c

    A_face = np.empty((num_triangles, num_triangles, 3, 3), np.complex128)
    phi_face = np.empty((num_triangles, num_triangles), np.complex128)

    # calculate all the integrations for each face pair
    for p in xrange(num_triangles): # p is the index of the observer face:
        nodes_p = nodes[triangle_nodes[p, :], :]
        for q in xrange(p): # q is the index of the source face, need for elements below diagonal

            nodes_q = nodes[triangle_nodes[q, :], :]
            if (any(triangle_nodes[p, :] == triangle_nodes[q, :])):
                # triangles have one or more common nodes, perform singularity extraction
                I_A, I_phi = core_for.face_integrals_smooth_complex(xi_eta_eval[None, :, :], weights, nodes_q, 
                                    xi_eta_eval, weights, nodes_p, jk_0)
        
                # the singular 1/R components are pre-calculated
                index_singular = scr_index(p, q, indices_precalc, indptr_precalc)

                I_A = I_A + A_precalc[index_singular, :, :]
                I_phi = I_phi + phi_precalc[index_singular]
        
            else:
                # just perform regular integration
                # As per RWG, triangle area must be cancelled in the integration
                # for non-singular terms the weights are unity and we DON't want to scale to triangle area
                I_A, I_phi = core_for.face_integrals_complex(xi_eta_eval, weights, nodes_q,
                                    xi_eta_eval, weights, nodes_p, jk_0)

            # by symmetry of Galerkin procedure, transposed components are identical (but transposed node indices)
            A_face[p, q, :, :] = I_A
            A_face[q, p, :, :] = I_A.T
            phi_face[p, q] = I_phi
            phi_face[q, p] = I_phi


    return A_face, phi_face

def scr_index(row, col, indices, indptr):
    """
    ! Convert compressed sparse row notation into an index within an array
    ! row, col - row and column into the sparse array
    ! indices, indptr - arrays of indices and index pointers
    ! NB: everything is assumed ZERO BASED!!
    ! Indices are not assumed to be sorted within the column
    """

    for n in xrange(indptr[row],indptr[row+1]):
        if indices[n]==col:
           return n
    return None # value not found, so return an error code

