import numpy as np
import os
import sys
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

def read_vtk(filename):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    polydata = reader.GetOutput()
    return polydata

def write_vtk(polydata, filename):
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    #triangulate
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(polydata)
    tri.Update()
    #writer.SetInputData(polydata)
    writer.SetInputData(tri.GetOutput())
    writer.Update()

def read_vtk_points(polydata):
    points = vtk_to_numpy(polydata.GetPoints().GetData())
    point_data = {}
    for i in range(polydata.GetPointData().GetNumberOfArrays()):
        name = polydata.GetPointData().GetArrayName(i)
        point_data[name] = vtk_to_numpy(polydata.GetPointData().GetArray(i))
    return points, point_data

def write_vtk_polydata(points, point_data, connectivity, cell_data={}, filename = 'medial.vtk'):
    vtk_pts = vtk.vtkPoints()
    for pt in points:
        vtk_pts.InsertNextPoint(pt)

    cells = vtk.vtkCellArray()
    for polygon_connectivity in connectivity:
        cell = vtk.vtkPolygon()
        cell.GetPointIds().SetNumberOfIds(len(polygon_connectivity))
        for i, point_index in enumerate(polygon_connectivity):
            cell.GetPointIds().SetId(i, point_index)
        cells.InsertNextCell(cell)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_pts)
    polydata.SetPolys(cells)
    vtk_point_data = polydata.GetPointData()

    for name in point_data:
        print(name)
        point_data_array = vtk.vtkDoubleArray()
        for value in point_data[name]:
            point_data_array.InsertNextValue(value)
        point_data_array.SetName(name)
        vtk_point_data.AddArray(point_data_array)
    
    write_vtk(polydata, filename)

def convert(inp_file, outfile):
    polydata = read_vtk(inp_file)
    points, point_data = read_vtk_points(polydata)
    indices, sides, slices = point_data['indices'].astype(int), point_data['side'].astype(int), point_data['slices'].astype(int)
    ids = np.arange(len(points))
    mapping = np.zeros([np.max(indices)+1,np.max(sides)+1,np.max(slices)+1],dtype=int) -1
    for i in range(len(ids)):
        mapping[indices[i],sides[i],slices[i]] = ids[i]

    med_points = []
    med_ids = np.zeros(len(points),dtype=bool)
    med_ids2 = np.zeros(len(points),dtype=int)
    ii = 0
    for i,pt in enumerate(points):
        if sides[i] == 0:
            med_ids[i] = True
            i2 = mapping[indices[i],1,slices[i]]
            med_points.append((pt+points[i2])/2.)
            med_ids2[i] = ii
            med_ids2[i2] = ii
            ii += 1
    med_points = np.array(med_points)

    med_point_data = {}
    for k,d in point_data.items():
        if k in ['Thickness', 'Radius', 'STJ', 'VAJ', 'IAS']:
            med_point_data[k] = d[med_ids]

    mapping = np.zeros([np.max(indices)+1,np.max(sides)+1,np.max(slices)+1],dtype=int) -1
    for i in range(len(ids)):
        mapping[indices[i],sides[i],slices[i]] = med_ids2[i]
    connectivity = []
    for i in range(len(mapping)-1):
        for j in range(1):
            for k in range(len(mapping[0,0])):
                connectivity.append([mapping[i,j,k-1], mapping[i+1,j,k-1],mapping[i+1,j,k],mapping[i,j,k]])

    write_vtk_polydata(med_points, med_point_data, connectivity, filename=out_file)

if __name__ == '__main__':
    inp_file = sys.argv[1]
    out_file = sys.argv[2]
    convert(inp_file, out_file)
