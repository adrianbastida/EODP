
from netCDF4 import Dataset
import numpy as np
import os
import sys
from common.io.mkdirOutputdir import mkdirOutputdir

def writeToa(outputdir, name, toa, quality_mask=None):

    # Check output directory
    mkdirOutputdir(outputdir)

    # TOA filename
    savetostr = os.path.join(outputdir, name + '.nc')

    # open a netCDF file to write
    ncout = Dataset(savetostr, 'w', format='NETCDF4')
    
    # define axis size
    ncout.createDimension('alt_lines', toa.shape[0])  # unlimited
    ncout.createDimension('act_columns', toa.shape[1])

    # create variable array
    floris_toa_scene = ncout.createVariable('toa', 'float32',
                                            ('alt_lines', 'act_columns',))

    # Assign data
    floris_toa_scene[:]         = toa[:]
    if quality_mask is not None:
        quality_mask = np.asarray(quality_mask, dtype=np.uint8)
        if quality_mask.shape != toa.shape:
            ncout.close()
            raise ValueError("La máscara de calidad debe tener la misma forma que TOA")
        quality = ncout.createVariable('quality', 'uint8', ('alt_lines', 'act_columns',))
        quality.long_name = 'Pixel quality flags'
        quality.flag_masks = np.array([1, 2, 4], dtype=np.uint8)
        quality.flag_meanings = 'bad_pixel dead_pixel saturated_pixel'
        quality[:] = quality_mask
    
    # close files
    ncout.close()

    print("Finished writting: " + savetostr)

def readQualityMask(directory, filename):
    ncfile = os.path.join(directory, filename)
    if not os.path.isfile(ncfile):
        return None
    with Dataset(ncfile) as dset:
        if 'quality' not in dset.variables:
            return None
        return np.asarray(dset.variables['quality'][:], dtype=np.uint8)

def readToa(directory, filename):

    # concatenate filename and check that it exists
    ncfile = os.path.join(directory, filename)
    if not os.path.isfile(ncfile):
        sys.exit('File not found ' +ncfile + ". Exiting.")
    print('Reading ' + ncfile)

    # Load dataset
    dset = Dataset(ncfile)

    # Extract data from NetCDF file
    toa = np.array(dset.variables['toa'][:])

    dset.close()

    print('Size of matrix ' + str(toa.shape))

    return toa
