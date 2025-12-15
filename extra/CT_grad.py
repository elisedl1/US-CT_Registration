import nrrd 
import numpy as np 


CT_grad_file = "/Users/elisedonszelmann-lund/Masters_Utils/Rivas_Data/CaninePhantom/co_registered/CT_grads.nrrd"
CT_seg_file = "/Users/elisedonszelmann-lund/Masters_Utils/Rivas_Data/CaninePhantom/co_registered/CT_seg.nrrd"
output_file = "/Users/elisedonszelmann-lund/Masters_Utils/Rivas_Data/CaninePhantom/co_registered/CT_grad_masked.nrrd"

CT_grad, grad_header = nrrd.read(CT_grad_file)
CT_seg, seg_header = nrrd.read(CT_seg_file)

mask = CT_seg > 0
CT_grad_masked = np.where(mask, CT_grad, 0)

nrrd.write(output_file, CT_grad_masked, grad_header)

