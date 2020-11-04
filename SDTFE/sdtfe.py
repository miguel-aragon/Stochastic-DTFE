#
#
#  Test point in tetrahedron using CGAL: https://doc.cgal.org/latest/Generator/index.html
#
#  http://steve.hollasch.net/cgindex/geometry/ptintet.html
#
#  https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
#
#

from pyhull.delaunay import DelaunayTri
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

#-----------------------------------------------------
# https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not
#-----------------------------------------------------
def sameside(v1,v2,v3,v4,p):
    normal = np.cross(v2-v1, v3-v1)
    dot1 = np.dot(normal, v4-v1)
    dot2 = np.dot(normal, p-v1)
    return (dot1 * dot2)  > 0

#-----------------------------------------------------
#
#-----------------------------------------------------
def tetraCoord(A,B,C,D):
    v1 = B-A ; v2 = C-A ; v3 = D-A
    # mat defines an affine transform from the tetrahedron to the orthogonal system
    mat = np.concatenate((np.array((v1,v2,v3,A)).T, np.array([[0,0,0,1]])))
    # The inverse matrix does the opposite (from orthogonal to tetrahedron)
    M1 = np.linalg.inv(mat)
    return (M1)

#-----------------------------------------------------
#
#-----------------------------------------------------
def pointInsideT(v1,v2,v3,v4,p):
    # Find the transform matrix from orthogonal to tetrahedron system
    M1=tetraCoord(v1,v2,v3,v4)
    # apply the transform to P
    p1 = np.append(p,1)
    newp = M1.dot(p1)
    # perform test
    return (np.all(newp>=0) and np.all(newp <=1) and sameside(v2,v3,v4,v1,p))

#-----------------------------------------------------
#
#-----------------------------------------------------
def tetra_volume(_a,_b,_c,_d):
    v1 = _b-_a
    v2 = _c-_a
    v3 = _d-_a
    
    return np.abs(( v1[:,0]*v2[:,1]*v3[:,2] + v1[:,1]*v2[:,2]*v3[:,0] + v1[:,2]*v2[:,0]*v3[:,1] - 
                   (v1[:,2]*v2[:,1]*v3[:,0] + v1[:,1]*v2[:,0]*v3[:,2] + v1[:,0]*v2[:,2]*v3[:,1]) ))/6.0


#-----------------------------------------------------
#
#-----------------------------------------------------
def adjacent_volumes(_xyz, _verts):

    #--- Compute volumes for all tetrahedron
    a = _xyz[_verts[:,0],:]
    b = _xyz[_verts[:,1],:]
    c = _xyz[_verts[:,2],:]
    d = _xyz[_verts[:,3],:] 
    tetra_volumes = tetra_volume(a,b,c,d)
    
    #--- Array to store adjacent volumes to each vertex
    adj_volumes = np.zeros(_xyz.shape[0])
    
    #--- Iterate over the volumes corresponding to the vertices array
    for i, tet_i in enumerate(_verts):
        #--- Iterate over the vertices in this tetrahedron and add its adjacent volume
        for ver_j in tet_i:
            adj_volumes[ver_j] = adj_volumes[ver_j] + tetra_volumes[i]
        
    return adj_volumes

#-----------------------------------------------------
#
#-----------------------------------------------------
def get_adjacent_voronoi(_xyz, _tetra):
    
    #--- Create empty list of lists
    adj_voro      = []
    for i in range(_xyz.shape[0]):
        adj_voro.append([])
    
    #--- Iterate over tetrahedra and assign each to its points
    for i, tet_i in enumerate(_tetra):

        #--- assign this tetrahedron to its vertices
        for j in range(4):        
            adj_voro[tet_i[j]].append(i)
    
    return adj_voro


#-----------------------------------------------------
#
#-----------------------------------------------------
def get_adjacent_voronoi_stats(_xyz, _tetra, _adj_voro):
    
    xt = np.zeros((2,len(_adj_voro)), dtype=np.float32)
    yt = np.zeros((2,len(_adj_voro)), dtype=np.float32)
    zt = np.zeros((2,len(_adj_voro)), dtype=np.float32)
    
    #--- Iterate over Voronoi cells
    for i, adj_i in enumerate(_adj_voro):
        
        xi = []
        yi = []
        zi = []
        #--- For this Voronoi cell iterate over its tetrahedra, tet_i is the index
        for tet_j in adj_i:
            
            #--- Retrieve this tetrahedron:
            tet_j = _tetra[tet_j,:]

            #--- These are the coordinates of the tetrahedron's vertices
            xi.extend(_xyz[tet_j,0])
            yi.extend(_xyz[tet_j,1])
            zi.extend(_xyz[tet_j,2])
        
        if len(xi) == 0:
            print(">>> ERROR ", i, adj_i)
            continue
    
        xt[0,i] = np.min(xi)
        xt[1,i] = np.max(xi)
        yt[0,i] = np.min(yi)
        yt[1,i] = np.max(yi)
        zt[0,i] = np.min(zi)
        zt[1,i] = np.max(zi)

    return [xt,yt,zt]


#-----------------------------------------------------
#
#-----------------------------------------------------
def is_in_adjacent_voronoi(_xyz, _tetra, _adj, _p):

    is_inside = 0
    
    #--- Iterate over adjacent voronoi cell
    for adj_i in _adj:
        
        #--- Retrieve this tetrahedron:
        tetra_i = _tetra[adj_i,:]

        #--- These are the coordinates of the tetrahedron's vertices
        xi = _xyz[tetra_i,0]
        yi = _xyz[tetra_i,1]
        zi = _xyz[tetra_i,2]

        v1 = np.asarray([xi[0], yi[0], zi[0]],dtype=np.float32)
        v2 = np.asarray([xi[1], yi[1], zi[1]],dtype=np.float32)
        v3 = np.asarray([xi[2], yi[2], zi[2]],dtype=np.float32)
        v4 = np.asarray([xi[3], yi[3], zi[3]],dtype=np.float32)
        
        if (pointInsideT(v1,v2,v3,v4, _p) == True):
            is_inside = 1
    
    return is_inside


#-----------------------------------------------------
#
#-----------------------------------------------------
def add_random_point_uniform(_xyz, _tet, _adj_voro, _adj_voro_box):
    
    xyz_new = _xyz.copy()
    
    n = _xyz.shape[0]
    #--- Loop over points
    for i, xyz_i in enumerate(_xyz):
        
        if (xyz_i[0] < 250 or xyz_i[0] > 1750): continue
        if (xyz_i[1] < 250 or xyz_i[1] > 1750): continue
        if (xyz_i[2] < 250 or xyz_i[2] > 1750): continue
        
        #--- Bounding box
        xb = _adj_voro_box[0][:,i]
        yb = _adj_voro_box[1][:,i]
        zb = _adj_voro_box[2][:,i]
                
        #--- Loop until a point inside the adjacent voronoi cell is found
        while(1):
            
            #--- Propose random point
            xr = np.random.uniform(xb[0], xb[1])
            yr = np.random.uniform(yb[0], yb[1])
            zr = np.random.uniform(zb[0], zb[1])
            
            p_i = np.asarray([xr,yr,zr])
            
            #--- Check if point lies inside adjacent Voronoi cell
            ins = is_in_adjacent_voronoi(_xyz, _tet, _adj_voro[i], p_i)
            
            if (ins == 1): 
                xyz_new[i,:] = p_i
                break
    
    return xyz_new


#-----------------------------------------------------
#
#-----------------------------------------------------
def add_random_point_normal(_xyz, _tet, _adj_voro, _adj_voro_vol, _gauss_scale=1):
    
    xyz_new = _xyz.copy()
    
    cnt1 = 3.0/(4.0*np.pi)
    cnt2 = 1.0/3.0
    
    n = _xyz.shape[0]
    #--- Loop over points
    for i, xyz_i in enumerate(_xyz):
        
        clear_output(wait=True)
                
        #--- Approximate radius
        rad_i = np.power(cnt1 * _adj_voro_vol[i], cnt2)
        
        #--- Scale radius by some factor
        rad_i = rad_i / _gauss_scale
        
        #--- Loop until a point inside the adjacent voronoi cell is found
        while(1):
            
            #--- Propose Gaussian random point
            xr = np.random.normal(0, rad_i)
            yr = np.random.normal(0, rad_i)
            zr = np.random.normal(0, rad_i)
            
            #--- Add random perturbation to point
            p_i = xyz_i + np.asarray([xr,yr,zr])
            
            #--- Check if point lies inside adjacent Voronoi cell
            #ins = is_in_adjacent_voronoi(_xyz, _tet, _adj_voro[i], p_i)
            
            xyz_new[i,:] = p_i
            break

         
            if (ins == 1): 
                xyz_new[i,:] = p_i
                break
        print(">>> ", i, " of ", n)
    
    return xyz_new


#-----------------------------------
#
#-----------------------------------
def write_cgal(xyz, boxsize, filename):

    npart = xyz.shape[0]

    #--- Define header
    h0 = np.array([boxsize], dtype='float32')
    h1 = np.array([npart], dtype='int32')

    #--- Binary write
    F = open(filename, "bw")

    #--- Write header to file
    h0.tofile(F)
    h1.tofile(F)

    #--- write volume data
    xyz.astype(dtype='float32').tofile(F)
    
    F.close()
    
#-----------------------------------------------------
#
#-----------------------------------------------------
def compute_sdtfe(xyz, box, n_ens, filebase, gauss_scale=3):

    #--- Compute Delaunay tessellation
    delau = DelaunayTri(xyz)
    tet = np.asarray(delau.vertices, dtype=np.int32)
    
    #--- Volume of adjacent Voronoi cell
    adj_voro_vol = adjacent_volumes(xyz, tet)
    
    #--- Get Adjacent Voronoi cell for all points
    adj_voro = get_adjacent_voronoi(xyz, tet)
    
    #--- Get bounding box of Voronoi cells
    adj_voro_box = get_adjacent_voronoi_stats(xyz, tet, adj_voro)

    #--- Generate ensemble
    for i in range(n_ens):
        
        print(">>> ", i)
        
        np.random.seed(i+1)
        new_xyz = add_random_point_normal(xyz, tet, adj_voro, adj_voro_vol, gauss_scale)

        write_cgal(new_xyz, box, filebase + str(i).zfill(3) + '.pos')
        np.save(filebase + str(i).zfill(3), new_xyz)




