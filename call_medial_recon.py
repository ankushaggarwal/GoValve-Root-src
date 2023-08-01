import sys
from medial_recon_from_bnd import *

outdir,tag,fnbnd_ref,fnmed_ref,nref,nstart,ndone = sys.argv[1:]
nref = int(nref)
nstart = int(nstart)
ndone = int(ndone)

for k in range(nstart,ndone):
    fnbnd = outdir + '/seg_' + tag + '_bnd_' + str(k)+ '.vtk'
    fnmed_out = outdir + '/seg_' + tag + '_med_recon_' + str(k) + '.vtk'
    if k == nref:
        continue
    convert(fnbnd,fnmed_out)
    print('Done with ' + str(k))