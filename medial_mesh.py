import numpy as np
from scipy.interpolate import splprep, splev, splrep
from scipy.ndimage import binary_closing, label, center_of_mass, gaussian_filter
from scipy.spatial import ConvexHull
import math
from skimage import measure
import vtk
from vtk.util.numpy_support import vtk_to_numpy

def rotz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])

def spline2d(x, y, npts):
    tck, u = splprep([x, y], s=0, per=True)
    u_new = np.linspace(u.min(), u.max(), npts)
    x_new, y_new = splev(u_new, tck)
    return x_new, y_new

def reorient(vtkfile):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtkfile)
    reader.Update()
    polydata = reader.GetOutput()
    labels = vtk_to_numpy(polydata.GetPointData().GetArray('Label'))
    STJ = labels==1
    VAJ = labels==2
    IAS = labels==6
    pts = vtk_to_numpy(polydata.GetPoints().GetData())
    outflow_vec1 = np.mean(pts[VAJ])
    outflow_vec2 = np.mean(pts[STJ])
    center = -np.mean(pts[VAJ],axis=0)
    #print(center)
    long_axis = np.mean(pts[STJ],axis=0)-np.mean(pts[VAJ],axis=0)
    long_axis = long_axis/np.linalg.norm(long_axis)

    ias_axis = np.mean(pts[IAS]+center - np.outer(np.dot(pts[IAS]+center,long_axis),long_axis),axis=0)
    ias_axis  = ias_axis/np.linalg.norm(ias_axis)

    third_axis = np.cross(long_axis,ias_axis)

    R = np.vstack((ias_axis,third_axis,long_axis))
    T = np.hstack((R,R@(center).reshape(-1,1)))
    T = np.vstack((T,np.array([0,0,0,1])))
    transform = vtk.vtkTransform()
    transform.SetMatrix(T.flatten())

    transformFilter=vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputConnection(reader.GetOutputPort())
    transformFilter.Update()
    #writer = vtk.vtkPolyDataWriter()
    #writer.SetInputConnection(transformFilter.GetOutputPort())
    #newfile = vtkfile.split('.')[0]+'-rotated.vtk'
    #print('writing ',newfile)
    #writer.SetFileName(newfile)
    #writer.Write()

    #Tinv = np.hstack((R.T,(-center).reshape(-1,1)))
    #Tinv = np.vstack((Tinv,np.array([0,0,0,1])))
    #transform_inv = vtk.vtkTransform()
    #transform_inv.SetMatrix(Tinv.flatten()) #or simply use the transform.GetLinearInverse as below

    #transformFilter2=vtk.vtkTransformPolyDataFilter()
    #transformFilter2.SetTransform(transform.GetLinearInverse()) #now this can be applied to any other mesh
    ##transformFilter2.SetTransform(transform_inv) 
    #transformFilter2.SetInputConnection(transformFilter.GetOutputPort())
    #transformFilter2.Update()
    #writer = vtk.vtkPolyDataWriter()
    #writer.SetInputConnection(transformFilter2.GetOutputPort())
    #newfile = vtkfile.split('.')[0]+'-rotated-back.vtk'
    #print('writing ',newfile)
    #writer.SetFileName(newfile)
    #writer.Write()
    
    return transform, vtk_to_numpy(transformFilter.GetOutput().GetPoints().GetData())

def edge_sampling(pts_rotxyz):
    nsamp = 48
    gamma = np.arange(0, 360, 10)
    pts_resamp = []
    indices, sides, slices = [], [], []
    # create a 50X50 matrix with one enteries in the circle
    s_mat = np.zeros((100, 100),dtype=int)
    for i in range(100):
        for j in range(100):
            if (i-50)**2 + (j-50)**2 <= 50**2:
                s_mat[i,j] = 1
    for I in range(len(gamma)):
        print(I)
        rot_ang = gamma[I]
        # Rotate the input point cloud about the z-axis
        pts_rot = np.dot(pts_rotxyz, rotz(np.deg2rad(rot_ang)))

        # Select a slab of points close to y = 0 along positive x-axis
        slice_ind = np.abs(pts_rot[:, 1]) < 1
        pts_slice = pts_rot[slice_ind, :]
        pts_slice = pts_slice[pts_slice[:, 0] > 0, :]

        # shift vertically so that all coordinates are positive
        buff_vertical = 10
        pts_slice[:, 2] = pts_slice[:, 2] + buff_vertical
        pts_slice = 10*pts_slice

        #find x and z extent and add a padding buffer
        buff_pad = 50
        x_rng = np.min(pts_slice[:, 0]), np.max(pts_slice[:, 0])
        z_rng = np.min(pts_slice[:, 2]), np.max(pts_slice[:, 2])
        x_rng = x_rng[0] - 0.1 * (x_rng[1] - x_rng[0]), x_rng[1] + 0.1 * (x_rng[1] - x_rng[0])
        z_rng = z_rng[0] - 0.1 * (z_rng[1] - z_rng[0]), z_rng[1] + 0.1 * (z_rng[1] - z_rng[0])
        

        # Pad the 2D slice and create a binary image
        img = np.zeros((math.ceil(np.max(pts_slice[:,2]) + buff_pad),math.ceil(np.max(pts_slice[:,0]) + buff_pad)),dtype=int)


        for j in range(pts_slice.shape[0]):
            x = pts_slice[j, 0]
            z = pts_slice[j, 2]
            img[int(z), int(x)] = 1

        # Perform morphological closing to smooth the edges
        closed_img = binary_closing(img, structure=s_mat)

        img_cl_sm = gaussian_filter(closed_img.astype('float'), 7)
        cuv = measure.find_contours(img_cl_sm,0.5)
        fap, ind, side = standardize(cuv[0].T,13)
        fap = fap/10.
        fap[:,0] = fap[:,0] - buff_vertical 
        new_pts = np.zeros([len(fap),3])
        new_pts[:,0], new_pts[:,2] = fap[:,1], fap[:,0]
        # Rotate back about the z-axis
        new_pts_rot = np.dot(new_pts, rotz(np.deg2rad(-rot_ang)))

        pts_resamp.append(new_pts_rot)
        indices.append(ind)
        sides.append(side)
        slices.append(np.ones(len(ind),dtype=int)*I)

    return np.array(pts_resamp).reshape(-1,3), np.array(indices).flatten(), np.array(sides).flatten(), np.array(slices).flatten()

#        # Find the contour points from the closed binary image
#        contour_pts = np.array(np.where(closed_img == 1)).T.astype(float)
#
#        # Compute the convex hull of the contour points
#        hull = ConvexHull(contour_pts)
#        contour_pts = contour_pts[hull.vertices, :]
#
#        # Ensure the contour points are in clockwise order
#        if hull.area > 0:
#            contour_pts = contour_pts[::-1, :]
#
#        # Resample the contour points using spline interpolation
#        x_resamp, z_resamp = spline2d(contour_pts[:, 0], contour_pts[:, 1], npts=200)
#        z_rng = np.min(z_resamp), np.max(z_resamp)
#        nqs = np.linspace(z_rng[0], z_rng[1], nsamp)
#        spline_ind = np.digitize(z_resamp, nqs)
#
#        # Reorder the indices based on the resampling to obtain the rotated and ordered indices
#        mi_ref_ord = np.arange(len(z_resamp))
#        mi_rot_ord = mi_ref_ord[spline_ind - 1]
#
#        # Transform the resampled points to the rotated and translated image
#        ind_rot_trans = np.array([x_resamp, np.zeros_like(x_resamp), z_resamp]).T
#        img_rot_trans = np.zeros((100, 100, 100))
#        img_rot_trans[tuple(ind_rot_trans.astype(int).T)] = 1
#        labels, num_components = label(img_rot_trans)
#
#        if num_components > 0:
#            # Calculate the center of mass of each connected component
#            com_rot_trans = np.array(center_of_mass(img_rot_trans, labels, range(1, num_components + 1))).T
#            com_rot_trans[:, 1] = com_rot_trans[:, 1] * (z_rng[1] - z_rng[0]) / 100 + z_rng[0]
#            com_rot_trans[:, 0:2] = com_rot_trans[:, 0:2] * (100 / img_rot_trans.shape[0]) - 50
#
#            # Inverse rotate the points to get the resampled 3D contour
#            inv_rot = rotz(np.deg2rad(-rot_ang))
#            com_trans = np.dot(inv_rot, com_rot_trans).T
#            pts_resamp.append(com_trans)
#
#    return pts_resamp#, x_resamp, z_resamp

def standardize(cuv, nqs):
    #check if the points are clockwise or counter clockwise
    area = 0
    for i in range(cuv.shape[1]-1):
        area + (cuv[0,i+1]-cuv[0,i])*(cuv[1,i]+cuv[1,i+1])/2.
    if area > 0:
        cuv = np.flip(cuv,1)

    # Find vertical min and max and reorder point indices
    emax_idx = np.argmax(cuv[0, :])
    emin_idx = np.argmin(cuv[0, :])

    # Reorder the points
    ncp = cuv.shape[1]
    cuv_tcs = np.roll(cuv.T[:-1], ncp - emax_idx-1, axis=0).T
    cuv_tcs = np.hstack((cuv_tcs, cuv_tcs[:, 0].reshape(-1, 1)))

    if emax_idx > emin_idx:
        emin_cuv_tcs = emin_idx + (ncp - emax_idx)
    else:
        emin_cuv_tcs = emin_idx - emax_idx

    emax_cuv_tcs = 0  # Initialize emax_cuv_tcs before using it

    # Determine c1, c2 indices on opposite sides of the root near the vertical mean
    mean_pt = np.mean(cuv_tcs, axis=1)
    d = (cuv_tcs[0, :] - mean_pt[0])
    dsqrt = np.sqrt(d * d)
    c1_cuv_tcs = np.argmin(dsqrt[emax_cuv_tcs:emin_cuv_tcs]) #+ emax_cuv_tcs
    c2_cuv_tcs = np.argmin(dsqrt[emin_cuv_tcs:]) + emin_cuv_tcs - 1

    def interp(x,pts):
        f, u = splprep([pts[0], pts[1]], u=x,s=0)
        a = np.linspace(x[0], x[-1], nqs)
        return splev(a,f)

    # Spline interpolation for each quadrant
    x1 = np.arange(c1_cuv_tcs, emin_cuv_tcs+1)
    pts = cuv_tcs[:, x1]
    #f1, u = splprep(cuv_tcs[:, x1],u=x1)
    #a1 = np.linspace(x1[0], x1[-1], nqs)
    #fap1 = splev(a1, f1)
    fap1 = interp(x1,pts)

    x2 = np.arange(emin_cuv_tcs, c2_cuv_tcs + 1)
    #f2, u = splprep(cuv_tcs[:, x2],u=x2)
    #a2 = np.linspace(x2[0], x2[-1], nqs)
    #fap2 = splev(a2, f2)
    pts = cuv_tcs[:, x2]
    fap2 = interp(x2,pts)

    x3 = np.arange(c2_cuv_tcs, ncp + 1)
    #f3, u = splprep(np.hstack((cuv_tcs[:, c2_cuv_tcs:ncp], cuv_tcs[:, 0:1])),u=x3)
    #a3 = np.linspace(x3[0], x3[-1], nqs)
    #fap3 = splev(a3, f3)
    pts = np.hstack((cuv_tcs[:, c2_cuv_tcs:ncp], cuv_tcs[:, 0:1]))
    fap3 = interp(x3,pts)

    x4 = np.arange(emax_cuv_tcs, c1_cuv_tcs + 1)
    #f4, u = splprep(cuv_tcs[:, x4],u=x4)
    #a4 = np.linspace(x4[0], x4[-1], nqs)
    #fap4 = splev(a4, f4)
    pts = cuv_tcs[:, x4]
    fap4 = interp(x4,pts)
    #print(np.array(fap2)[:,1:],np.array(fap3)[:,1:],np.array(fap4)[:,1:-1])
    # Resampled 2D contour with all four quadrants
    #fap = np.hstack((np.array(fap1),np.array(fap2)[:,1:],np.array(fap3)[:,1:],np.array(fap4)[:,1:-1])).T
    fap1 = np.array(fap1).T
    fap2 = np.array(fap2).T
    fap3 = np.array(fap3).T
    fap4 = np.array(fap4).T
    #fap = np.vstack((fap2,fap3[1:],fap4[1:],fap1[1:-1]))
    fap = np.vstack((fap2,fap3[1:],fap4[:],fap1[1:]))
    index = np.arange(len(fap2))
    ind2 = index.copy()
    ind1 = ind2[::-1]
    ind3 = index + len(ind2)-1
    ind4 = ind3[::-1]
    #indices = np.concatenate((ind2,ind3[1:],ind4[1:],ind1[1:-1]))
    indices = np.concatenate((ind2,ind3[1:],ind4[:],ind1[1:]))
    #side = np.concatenate((np.zeros(len(fap2)*2-2),np.ones(len(fap2)*2-2)))
    side = np.concatenate((np.zeros(len(fap2)*2-1),np.ones(len(fap2)*2-1)))

    return fap, indices.astype(int), side.astype(int)

T_init, pts_rotxyz = reorient('/Users/ankushaggarwal/Downloads/ticket_5/segref-reoriented.vtk')

pts, indices, sides, slices = edge_sampling(pts_rotxyz)

ids = np.arange(len(pts),dtype=int)
mapping = np.zeros([np.max(indices)+1,np.max(sides)+1,np.max(slices)+1],dtype=int) -1
for i in range(len(ids)):
    mapping[indices[i],sides[i],slices[i]]=ids[i]
connectivity = []
for i in range(len(mapping)-1):
    for j in range(len(mapping[0])):
        for k in range(len(mapping[0,0])):
            connectivity.append([mapping[i,j,k-1], mapping[i+1,j,k-1],mapping[i+1,j,k],mapping[i,j,k]])
thickness = np.zeros(len(pts))
for i in range(len(mapping)):
    for k in range(len(mapping[0,0])):
        i1, i2 = mapping[i,0,k], mapping[i,1,k]
        dis = np.linalg.norm(pts[i1]-pts[i2])
        if dis<1e-3:
            dis = 0.3
        thickness[i1], thickness[i2] = dis, dis

points = vtk.vtkPoints()
for pt in pts:
    points.InsertNextPoint(pt)
cells = vtk.vtkCellArray()
for polygon_connectivity in connectivity:
    cell = vtk.vtkPolygon()
    cell.GetPointIds().SetNumberOfIds(len(polygon_connectivity))
    for i, point_index in enumerate(polygon_connectivity):
        cell.GetPointIds().SetId(i, point_index)
    cells.InsertNextCell(cell)
poly_data = vtk.vtkPolyData()
poly_data.SetPoints(points)
poly_data.SetPolys(cells)
point_data = poly_data.GetPointData()

point_data_array = vtk.vtkDoubleArray()
for value in indices:
    point_data_array.InsertNextValue(value)
point_data_array.SetName("indices")
point_data.SetScalars(point_data_array)

point_data_array = vtk.vtkDoubleArray()
for value in sides:
    point_data_array.InsertNextValue(value)
point_data_array.SetName("side")
point_data.AddArray(point_data_array)

point_data_array = vtk.vtkDoubleArray()
for value in slices:
    point_data_array.InsertNextValue(value)
point_data_array.SetName("slices")
point_data.AddArray(point_data_array)

point_data_array = vtk.vtkDoubleArray()
for value in thickness:
    point_data_array.InsertNextValue(value)
point_data_array.SetName("Thickness")
point_data.AddArray(point_data_array)

point_data_array = vtk.vtkDoubleArray()
for value in thickness:
    point_data_array.InsertNextValue(value/2.)
point_data_array.SetName("Radius")
point_data.AddArray(point_data_array)

point_data_array = vtk.vtkDoubleArray()
for value in indices:
    if value == 0:
        point_data_array.InsertNextValue(1)
    else:
        point_data_array.InsertNextValue(0)
point_data_array.SetName("STJ")
point_data.AddArray(point_data_array)

point_data_array = vtk.vtkDoubleArray()
for value in indices:
    if value == np.max(indices):
        point_data_array.InsertNextValue(1)
    else:
        point_data_array.InsertNextValue(0)
point_data_array.SetName("VAJ")
point_data.AddArray(point_data_array)

point_data_array = vtk.vtkDoubleArray()
for value in slices:
    if value == 0:
        point_data_array.InsertNextValue(1)
    else:
        point_data_array.InsertNextValue(0)
point_data_array.SetName("IAS")
point_data.AddArray(point_data_array)

#triangulate
tri = vtk.vtkTriangleFilter()
tri.SetInputData(poly_data)
tri.Update()

output_filename = "output_mesh.vtk"
#writer = vtk.vtkPolyDataWriter()
#writer.SetFileName(output_filename)
#writer.SetInputData(tri.GetOutput())
#writer.Write()

transformFilter2=vtk.vtkTransformPolyDataFilter()
transformFilter2.SetTransform(T_init.GetLinearInverse()) #now this can be applied to any other mesh
#transformFilter2.SetTransform(transform_inv) 
transformFilter2.SetInputConnection(tri.GetOutputPort())
transformFilter2.Update()
writer = vtk.vtkPolyDataWriter()
writer.SetFileName(output_filename)
writer.SetInputConnection(transformFilter2.GetOutputPort())
writer.Write()