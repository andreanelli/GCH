import argparse
import numpy as np
import os
import shutil
import json
# Make it work for Python 2+3 and with Unicode
import io

try:
    to_unicode = unicode
except NameError:
    to_unicode = str


from lib_gch import *
from gch_utils import *

def gch_init(pk,pnrg,setxyz,wdir_local,s_c,s_e,ndim,numref,numshaken,conv,mode):
    """ Given the kernel matrix, the structure's energies and their corresponding xyz dataset,
    it creates a subfolder in the working directory containing a set of
    structures shaken according to the desired uncertainty.The result
    will be a file called shaketraj.xyz.
    version 0.1 July 3rd 2018
    """

    # Building the pfile from the kernel using a kPCA and
    # passing the energy vector
    print("Loading the kernel matrix, it can take a minute if thousands of elements")
    pkern        = np.loadtxt(pk)
    energy       = np.loadtxt(pnrg)
    # Checks if energies and kernel are compatible
    assert (len(energy) >= pkern.shape[0]),"Colder than absolute zero!"

    ## Try to remove wdir directory in case it's there
    cwd = os.getcwd()
    wdir = cwd + '/' + wdir_local
    print(wdir)
    try:
        shutil.rmtree(wdir)
        os.mkdir(wdir)
    except:
        os.mkdir(wdir)

    # Builds an energy + kpca matrix [en kp_1 kp_2 kp_3 .... kp_npca]
    # To be refined into a more compact form
    npca         = 32
    pfile        = np.ones((len(energy),npca+1))
    pfile[:,0]   = energy
    pfile[:,1:]  = kpca(pkern,npca)
    cdir         = os.getcwd()
    # Preparing the input json for the gch
    pxyz         = setxyz
    wdir         = wdir
    sigma_c      = s_c # fractional uncertainty in DIFFERENCES in lattice vectors BETWEEN STRUCTURES
    sigma_e      = s_e # uncertainty in absolute energies
    nref         = numref # number of reference struct for estimation of uncertainty in KPCA descriptors
    nshaken      = numshaken # number of shaken struct per reference for estimation of uncertainty in KPCA descriptors
    inrg         = 0 # number of column containing energies in icedata
    cols         = np.arange(ndim) # numbers of columns to be used for GCH construction
    convth       = conv # limit on accuracy of probabilities of stabilisation (determines number of sampled GCHs)

    print ('DONE: Loaded data')
    print ('Initializing statistical sampling of the fuzzy GCH')
    #refids = initialize_sample_GCH(pfile,pxyz,sigma_c,nref,nshaken,wdir,inrg,cols)
    if mode=="random":
        initialize_random_sample_GCH(pfile,pxyz,sigma_c,nref,nshaken,wdir,inrg,cols)
        #initialize_random(pfile,pxyz,sigma_c,nref,nshaken,wdir,inrg,cols)
    if mode=="fps":
        fpsidx = FPS(pkern,nref)
        initialize_fps_sample_GCH(pfile,fpsidx,pxyz,sigma_c,nref,nshaken,wdir,inrg,cols)

    input_for_gchrun = {
        'wdir': wdir,
        'ref_kernel':cdir+'/'+pk ,
        'pnrg' : cdir+'/'+pnrg,
        'setxyz' : cdir+'/'+setxyz,
        'sigma_c': sigma_c,
        'sigma_e': sigma_e,
        'nref' : nref,
        'nshaken':nshaken,
        'inrg':inrg,
        'ndim' : ndim,
        'convergence_threshold': convth
        }

    #Save a bunch of stuff useful for later
    print("DONE ! go to "+wdir+"/ to see what's in there")
    # Writing of the parameters line, it will become a json file to be given as an input to the gch-run
    # in the next version

    with io.open(wdir+'/input.json', 'w', encoding='utf8') as outfile:
        str_ = json.dumps(input_for_gchrun,
                          indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(str_))

    np.save(wdir+'/nrg-proj',pfile)

    label = "Input parameters for gch are : wdir = "+wdir+" , sigma_c = {} , sigma_e = {} , nref = {},\
     nshaken = {}, inrg = {}, ndim = {}, convergence_threshold = {} \
     ".format(sigma_c,sigma_e,nref,nshaken,inrg,cols,convth)
    fo = open(wdir+"/labels.dat", "a")
    fo.write( label )
    fo.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A code that prepares the dataset for running a \
    generalised convex hull construction. It requires as input a dataset, a kernel measure of its\
    structures' similarity and their energies)")
    parser.add_argument("kernel", help="Kernel file")
    parser.add_argument("-nrg", help="Molar energy file, has to be as long as the number of rows of kernel")
    parser.add_argument("-ixyz", help="libatoms formatted xyz file of dataset")
    parser.add_argument("-wdir",type=str,default='./',help="Directory where to put the shaken subfolder")
    parser.add_argument("-sc",type=float,default=0.001,help="Cartesian uncertainty in units of measure (default is 1e-3). n.b. Referred to the units used in the dataset (default Angstrom).")
    parser.add_argument("-se",type=float,default=0.01,help="Energetic uncertainty in units of measure (default is 1e-2). n.b. Referred to the units used for the energies (default meV).")
    parser.add_argument("--ndim",type=int,default=2,help="Specify the dimensionality for hull construction, default is 1 dimensional (E vs KPCA1)")
    parser.add_argument("--nref",type=int,default=50,help="Number of references to be extracted to build the uncertainties on (default 50 structures, 100 is a good guess in general")
    parser.add_argument("--nshake",type=int,default=50,help="Number of shaken repetitions on ref structures (default 50, it's plenty already)")
    parser.add_argument("--conv",type=float,default=0.25,help="Number of samples hulls to build, given by 1/conv ( default is 0.25, corresponding to 400 hulls)")
    parser.add_argument("--mode",type=str,default="random",help="Selection criteria for points to be shaken : \
    ['random' for random choice or 'fps' for a farthest point sampling based choice ] (Default random, use fps for sparser sampling) ")



    args = parser.parse_args()

    gch_init(args.kernel,args.nrg,args.ixyz,args.wdir,args.sc,args.se,args.ndim,args.nref,args.nshake,args.conv,args.mode)
