import argparse
import numpy as np
import os
import shutil
import json
# Make it work for Python 2+3 and with Unicode
import io


from gch_utils import *
from lib_gch import *
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

def gch_run(shk,wdir,mp):
    """ "A code that runs the generalised convex hull construction\
    using the dataset constructed with gch-init.py. It requires in input all the input used in gch-init \
    and the out of sample kernel of the shaken points used for estimating the kernel structural response.\
    It will read the input from the folder used as 'wdir'."  """
    # Read all the paramters from the json input_for_gchrun
    with open(wdir+'/input.json') as data_file:
        input_for_gchrun = json.load(data_file)

    print "We will load both the dataset kernel and the shaken kernel, it could take some minutes in case of thousands of structures.."
    pfile      = np.load(wdir+'/nrg-proj.npy')
    # I know it looks dodgy, but to make it easy parallelisable
    # we want pfile to be global
    
    pxyz       = input_for_gchrun["setxyz"]
    refids     = np.loadtxt(wdir+'/refstruct.idx',dtype='int')
    refkernel  = np.loadtxt(input_for_gchrun["ref_kernel"])
    ooskernel  = np.loadtxt(shk)
    shakenproj = ookpca(refkernel,ooskernel,32)

    np.save(wdir+'/shaketraj',shakenproj)
    sigma_c    = input_for_gchrun["sigma_c"] # fractional uncertainty in DIFFERENCES in lattice vectors BETWEEN STRUCTURES
    sigma_e    = input_for_gchrun["sigma_e"] # uncertainty in absolute energies
    nref       = input_for_gchrun["nref"] # number of reference struct for estimation of uncertainty in KPCA descriptors
    nshaken    = input_for_gchrun["nshaken"] # number of shaken struct per reference for estimation of uncertainty in KPCA descriptors
    inrg       = input_for_gchrun["inrg"] # number of column containing energies in icedata
    cols       = range(input_for_gchrun["ndim"]) # numbers of columns to be used for GCH constructions
    convth     = input_for_gchrun["convergence_threshold"] # limit on accuracy of probabilities of stabilisation (determines number of sampled GCHs)
    minprob    = mp
    N = int(100./convth)
    print "You have selected "+str(N)+" convex hulls samples per pruning iterations"
    #Npp = int(round(N/nproc))
    #rank = range(nproc)
    #Niter = nit
    print 'Statistical sampling of the fuzzy GCH'

    # statistical sampling of the fuzzy GCH
    # spam the nproc processes !
    vprobprune = prune_GCH(pfile,sigma_e,convth,refids,nshaken,wdir,inrg,cols,minprob,restart=True)
    #vprobprune = parallel_prune_GCH(pfile,refids[0:nref],nshaken,wdir,500,inrg,cols,nproc,convth)

    np.savetxt(wdir+'/vprobprune.dat',vprobprune)
    np.savetxt(wdir+'/vlist.idx',np.where(vprobprune[-1]>0)[0],fmt='%i')
    # The file saved will be VPROBPRUNE, a list of probabilities of each structures of being a vertex,
    # at every iteration of the pruning procedure.

    print 'DONE: Statistical sampling of the fuzzy GCH'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A code that runs the generalised convex hull construction\
    using the dataset constructed with gch-init.py. It requires in input all the input used in gch-init \
    and the out of sample kernel of the shaken points used for estimating the kernel structural response.\
    It will read the input from the folder used as 'wdir'.")
    parser.add_argument("shaken_kernel", help="Kernel file")
    parser.add_argument("-wdir",type=str,default='./',help="Directory where to put the shaken subfolder")
    parser.add_argument("-minprob",type=float,default='.51',help="The pruning iterations will keep removing \
    structures from the set until every point has at least minprob probability of being a hullpsoint. Default to 51%, higher probabilities mean a finer selection" )



    args = parser.parse_args()

    gch_run(args.shaken_kernel,args.wdir,args.minprob)
