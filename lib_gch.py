import numpy as np
import random
from sklearn.linear_model import Ridge as ridgereg
import time
from scipy.spatial.distance import pdist,squareform
from scipy.linalg import eigh as EIGH
from scipy.spatial import ConvexHull as chull
import linecache
import os
from ase.io import read,write
import time
from scipy.spatial import ConvexHull as chull
from scipy.interpolate import griddata as sing
from scipy.special import erfcinv,erfinv
import multiprocessing


## ============================================================================
##                      GCH core functions
## ============================================================================



def get_refgch(pfile, inrg=0, cols=[]):
    """
    [[pfile]] : matrix : energy + kpca
    ndim               : number of dimensions of the low dimensional hull we use as reference
    inrg               : index of energy column
    [cols ] : list     : list of indices of columns of pfile to be used for the LD hull construction
    """

    data = pfile[:,cols]
    ndim = len(cols)
    nresdim = len(pfile[0,ndim::])
    hull = chull(data)


    HD_kpca = pfile[:,ndim::]
    snormals = hull.equations[:,0:ndim]
    sshifts  = hull.equations[:,ndim]

    vlist = hull.vertices
    if inrg >= 0 :  # discards points that face down in the given direction
        slist = hull.simplices
        vlist = np.asarray([], int)

        for i in xrange(len(slist)):
            if snormals[i,inrg] < 0.:
                vlist = np.union1d(vlist, slist[i])

    ns = len(sshifts)
    nd = len(data)
    ss = np.zeros((nd),dtype='int')

    projs = np.zeros((nd,ndim-1),dtype='float')
    interp_HDkpca = np.zeros((nd,pfile.shape[1]-ndim),dtype='float')

    elist = np.ones((len(data),2))*1e100
    sigma_s = np.zeros((nd,nresdim),dtype='float') ### Structural variance in the higher dimensions
    #sigma_s = np.zeros((nd),dtype='float') ### Structural variance in the higher dimensions

    for i in xrange(ns): # this is a loop over the simplices forming the boundary of the hull
        inrm = snormals[i]
        # if we are computing a purely structural chull (no energy column given)
        # or if we are on the downward-looking energy face of the chull (inrm[inrg]<0)
        # then we must consider this facet
        if inrg<0 or inrm[inrg] < 0.:

            for j in xrange(len(data)):
                dij = -(np.dot(inrm,data[j])+sshifts[i])  # this is the (signed) distance of the point j from the i-th simplex
                if np.abs(dij) < 1e-8: dij = 0
                if elist[j,0]>dij:
                    elist[j,0] = dij
                if inrg>=0:  # this is the vertical distance from the simplex along the energy direction
                    dz =  -dij/inrm[inrg]

                    if elist[j,1]>dz:
                        elist[j,1]=dz
                        ss[j] = i
                        projs[j] = data[j,1::]
    cntr = elist[:,1]

    for at in xrange(len(data)):
        if at in vlist :
            sigma_s[at] = 0.

        else :
            interp_HDkpca[at] = sing(data[hull.simplices[ss[at]],1::],HD_kpca[hull.simplices[ss[at]]],projs[at],method='linear')
            sigma_s[at] = HD_kpca[at]-interp_HDkpca[at]
            #sigma_s[at] = np.linalg.norm(HD_kpca[at]-interp_HDkpca[at])

    # normalize sigma_si

    if(sigma_s.any()!=0.0):
        sigma_s /= max(np.linalg.norm(sigma_s,axis=0))

    #sigma_s /= max(sigma_s)

    return vlist,cntr,sigma_s

def get_gch(pfile, inrg=0, cols=[]):
    """
    [[pfile]] : matrix : energy + kpca
    ndim               : number of dimensions of the low dimensional hull we use as reference
    inrg               : index of energy column
    [cols ] : list     : list of indices of columns of pfile to be used for the LD hull construction
    """
    data = pfile[:,cols]
    ndim = len(cols)
    nresdim = len(pfile[0,ndim::])
    hull = chull(data)

    HD_kpca = pfile[:,ndim::]
    snormals = hull.equations[:,0:ndim]

    vlist = hull.vertices
    if inrg >= 0 :  # discards points that face down in the given direction
        slist = hull.simplices
        vlist = np.asarray([], int)

        for i in xrange(len(slist)):
            if snormals[i,inrg] < 0.:
                vlist = np.union1d(vlist, slist[i])

    nd = len(data)

    projs = data[:,1::]
    interp_HDkpca = np.zeros((nd,pfile.shape[1]-ndim),dtype='float')

    elist = np.ones((len(data),2))*1e100
    sigma_s = np.zeros((nd,nresdim),dtype='float') ### Structural variance in the higher dimensions
    #sigma_s = np.zeros((nd),dtype='float') ### Structural variance in the higher dimensions

    for at in xrange(len(data)):
        if at in vlist :
            sigma_s[at] = 0.

        else :
            interp_HDkpca[at] = sing(data[vlist,1::],HD_kpca[vlist],projs[at],method='linear')
            sigma_s[at] = HD_kpca[at]-interp_HDkpca[at]
            #sigma_s[at] = np.linalg.norm(HD_kpca[at]-interp_HDkpca[at])

    # normalize sigma_s
    if(sigma_s.any()!=0.0):
        sigma_s /= max(np.linalg.norm(sigma_s,axis=0))

    return vlist,sigma_s

def eval_sampled_sigmaKPCA(refids,nsamples,wdir):
    """
    [[pfile]] : mat   : energy + kpca
    [refids] : list   : ids of reference sample structures from which
                        shaken structures were generated
    nsamples : scalar : number of shaken structures per reference
    wdir : string     : path to directory for saving shaken structs
    """

    # kpcaref = pfile[refids,1::]
    kpcaoos = np.load(wdir + '/shaketraj.npy')

    # initialize varKPCA
    varKPCA = kpcaoos[0,:] * 0.
    for iref in xrange(len(refids)) :
        samplesi = iref * (nsamples+1)
        samplesf = (iref + 1) * (nsamples+1) - 1
        #dkpca = kpcaoos[samplesi+1:samplesf,:] - kpcaref[iref,:]
        dkpca = kpcaoos[samplesi+1:samplesf,:] - kpcaoos[samplesi,:]
        var = np.var(dkpca,axis=0)
        varKPCA += var

    sigmaKPCA = np.sqrt(varKPCA/len(refids))

    return sigmaKPCA

def estimate_residual_sigmaE(pfile):
    """
    [[pfile]] : mat : energy + kpca
    [cols ] : list  : list of indices of columns of pfile to be used for the LD hull construction
    """

    # estimate energy response to all KPCA descriptors using ridge regression
    rr = ridgereg()
    rr.fit(pfile[:,1::],pfile[:,0])
    epsilon = rr.coef_

    return epsilon


## ============================================================================
##                      GCH initialisation functions
## ============================================================================


def create_samples_sigmaKPCA(pxyz,sigma_cell,refstructids,nsamples,wdir):
    """
    * needs ASE
    pxyz : string         : path to the xyz-dataset
    sigma_pos : scalar    : absolute (!) uncertainty in cartesian coordinates
    sigma_cell : scalar   : fractional (!) uncertainty in lattice vectors
    [refstructids] : list : list of reference structures to be randomised for
                            estimation of sigmaKPCA (uncertainty on structural descriptors)
    nsamples : scalar     : number of sample randomised structures per reference structure
    wdir : string         : path to directory for saving shaken structs
    """

    #dbqorig = quippy.AtomsList(pxyz)
    dbaorig = read(pxyz,index=':')

    dba = [dbaorig[int(iref)] for iref in refstructids]
    #We save the refstruct.idx as a sanity check
    np.savetxt(wdir + '/refstruct.idx',refstructids,fmt='%i')

    # evaluate sensible uncertainty in atomic positions given an uncertainty in the cell parameters
    vav = np.average( [ dba[n_ref].get_volume()/dba[n_ref].get_number_of_atoms() for n_ref in xrange(len(dba)) ] )
    sigma_pos = sigma_cell * np.cbrt(vav)

    print 'Uncertainty in Cartesian positions',sigma_pos

    shaketrajfull = []
    for n_ref in xrange(len(dba)):
        shaketraj = []
        shaketraj.append(dba[n_ref])
        shaketrajfull.append(dba[n_ref])

        for n_sample in xrange(nsamples) :

            shakeat = dba[n_ref].copy()

            # jitter cell
            cell = shakeat.cell
            ran = np.random.normal(np.zeros((3,3)),sigma_cell)
            cell += cell * ran
            shakeat.set_cell(cell,scale_atoms=True)

            # jitter atoms
            shakeat.rattle(sigma_pos,n_sample)

            shaketraj.append(shakeat)
            shaketrajfull.append(shakeat)

        # write trajectory of shaken structs corresponding to reference structure n_ref to file
        write(wdir + '/shaketraj.' + str(n_ref) + '.xyz',shaketraj)

    # write trajectory of shaken structs to file
    write(wdir + '/shaketraj.xyz',shaketrajfull)

    return

def initialize_random_sample_GCH(pfile,pxyz,sigma_cell,nref,nshaken,wdir,inrg=0,cols=[]):
    """
    [[pfile]] : mat     : energy + kpca
    pxyz : string       : path to the xyz-dataset
    sigma_cell : scalar : fractional uncertainty in lattice vectors
    nref : scalar       : number of reference structures for sampling of KPCA uncertainties
    nshaken : scalar    : number of shaken structures per reference
    wdir : string       : working directory for generation of rattled structure for
                          sampling of KPCA uncertainties
    inrg : scalar       : index of energy column in pfile
    [cols] : list       : list of indices of KPCA descriptors to be used in GCH construction
    * needs numpy.special.erfinv
    * note that embedding (nref * nshaken) structures OOS quickly requires a whole lot of memory
    """
    import random

    ien = inrg
    columns = cols

    # prepare reduction of dataset by constructing reference GCH
    v,contour,s = get_refgch(pfile,ien,columns)

    refids = random.sample(range(len(pfile)),nref)
    refids = np.array([int(rid) for rid in refids])

    # prepare sample structures for estimate uncertainty in KPCA descriptors
    # for jittering in sampling fuzzy GCH
    create_samples_sigmaKPCA(pxyz,sigma_cell,refids,nshaken,wdir)

    return refids

def initialize_fps_sample_GCH(pfile,fps_idx,pxyz,sigma_cell,nref,nshaken,wdir,inrg=0,cols=[]):
    """
    [[pfile]] : mat     : energy + kpca
    pxyz : string       : path to the xyz-dataset
    sigma_cell : scalar : fractional uncertainty in lattice vectors
    nref : scalar       : number of reference structures for sampling of KPCA uncertainties
    nshaken : scalar    : number of shaken structures per reference
    wdir : string       : working directory for generation of rattled structure for
                          sampling of KPCA uncertainties
    inrg : scalar       : index of energy column in pfile
    [cols] : list       : list of indices of KPCA descriptors to be used in GCH construction
    * needs numpy.special.erfinv
    * note that embedding (nref * nshaken) structures OOS quickly requires a whole lot of memory
    * note that it needs to have the fps index files stored in the wdir
    """
    import random

    ien = inrg
    columns = cols

    # prepare reduction of dataset by constructing reference GCH
    v,contour,s = get_refgch(pfile,ien,columns)

    #refids = random.sample(range(len(pfile)),nref)
    refids = fps_idx

    [int(rid) for rid in refids]

    # prepare sample structures for estimate uncertainty in KPCA descriptors
    # for jittering in sampling fuzzy GCH
    create_samples_sigmaKPCA(pxyz,sigma_cell,refids,nshaken,wdir)

    return refids


## ============================================================================
##                      GCH SAMPLING ROUTINES
## ============================================================================


def sample_GCH(pfile,sigma_ev,sigma_etot,epsilon,sigma_s,sigma_KPCA,convthresh,refids,nshaken,wdir,inrg,cols):
    """
    [[pfile]] : mat     : energy + kpca
    sigma_ev : scalar   : DFT uncertainty in total/absolute energies
    convthresh : scalar : measure of smallest vertex probabilities to be resolved
    [refids] : list     : ids of reference structures for sampling of KPCA uncertainties
    nshaken : scalar    : number of shaken structures per reference
    wdir : string       : working directory for generation of rattled structure for
                          sampling of KPCA uncertainties
    inrg : scalar       : index of energy column in pfile
    [cols] : list       : list of indices of KPCA descriptors to be used in GCH
                          construction
    """

    ndim = len(cols)
    nresdim = len(pfile[0,ndim::])

    # calculate number of GCHs to be sampled
    N = int(100./convthresh)

    # initialize vertex scores for reduced dataset at zero
    vertex_scores = np.zeros((len(pfile)),dtype='int')
    vertex_list = np.zeros((len(pfile),N),dtype='int')
    vertex_prob = np.zeros((len(pfile)),dtype='float')
    vertex_prob_prev = np.zeros((len(pfile)),dtype='float')

    # sample GCH
    # every candidate with probability>convthresh should have come up
    # around 100 times leaving the remnant uncertainty of the order of 1%

    for n in xrange(N):
        ## draw stabilities for all structures (within threshold of reference GCH) from Gaussian distr

        # update umcertainty in nrg according to previous GCH

        # Here:
        # -- dEDFT is the DFT error in energy
        # -- epsilon_i = RMS( dE/d\phi_i ) measures the typical energy
        #    response to variation of KPCA component i,
        # -- stdev(E) is the standard deviation in DFT energies across the
        #    dataset (as a measure of the overall energy response to all
        #    KPCA descriptors)
        # -- s = |{\bf s}| is the interpolatability/independence score and
        #    s_i measures the distance of a given structure X from the ideally
        #    interpolated counterpart X_GCH alond the i-th KPCA component (i>n)
        sigma = np.sqrt ( np.sum(np.square(sigma_s * epsilon),axis=1) ) / sigma_etot * sigma_ev

        # randomise nrg and kpca according to updated uncertainties
        # EAE : we only really need to update the kpca descriptors used for the GCH construction --> QUICKER
        nrg = np.random.normal(pfile[:,0],sigma[:])
        kpc = np.zeros((pfile[:,1::].shape))
        kpc = np.random.normal(kpc,1)
        kpc *= sigma_KPCA
        kpc += pfile[:,1::]

        # update input for GCH with randomised nrg
        tmp_pfile = np.column_stack((nrg,kpc))

        # construct new GCH for updated/randomised nrg
        v,sigma_s = get_gch(tmp_pfile,inrg,cols)
        vertex_scores[v] += 1
        vertex_list[v,n] = 1

        # evaluate probabilities r_vertex_prob based on r_vertex_scores/n
        #r_vertex_prob = r_vertex_scores*1./n
        vertex_prob = vertex_scores*1./(n+1)

        if ( (n+1)%200 == 0 ) :
            print "Iteration : ",n+1," in ",N

    return vertex_prob,vertex_list

def parallel_sample_GCH(rrank):
    """
    [[pfile]] : mat     : energy + kpca
    sigma_ev : scalar   : DFT uncertainty in total/absolute energies
    nstart   : scalar   : number of first sample GCH to be generated
    nfinish  : scalar   : number of final sample GCH to be generated

    [refids] : list     : ids of reference structures for sampling of KPCA uncertainties
    nshaken : scalar    : number of shaken structures per reference
    wdir : string       : working directory for generation of rattled structure for
                          sampling of KPCA uncertainties
    inrg : scalar       : index of energy column in pfile
    [cols] : list       : list of indices of KPCA descriptors to be used in GCH
                          construction
    """

    nstart =  rrank*Npp
    nfinish = nstart + Npp - 1
    np.random.seed(nstart)

    ndim = len(cols)
    nresdim = len(r_pfile[0,ndim::])

    # calculate number of GCHs to be sampled
    #N = int(100./convthresh)

    # initialize vertex scores for reduced dataset at zero
    vertex_scores = np.zeros((len(r_pfile)),dtype='int')
    #vertex_list = np.zeros((len(r_pfile),Npp),dtype='int')
    vertex_prob = np.zeros((len(r_pfile)),dtype='float')
    vertex_prob_prev = np.zeros((len(r_pfile)),dtype='float')

    # sample GCH
    # every candidate with probability>convthresh should have come up
    # around 100 times leaving the remnant uncertainty of the order of 1%

    for n in xrange(nstart,nfinish):
        ## draw stabilities for all structures (within threshold of reference GCH) from Gaussian distr

        # update umcertainty in nrg according to previous GCH

        # Here:
        # -- dEDFT is the DFT error in energy
        # -- epsilon_i = RMS( dE/d\phi_i ) measures the typical energy
        #    response to variation of KPCA component i,
        # -- stdev(E) is the standard deviation in DFT energies across the
        #    dataset (as a measure of the overall energy response to all
        #    KPCA descriptors)
        # -- s = |{\bf s}| is the interpolatability/independence score and
        #    s_i measures the distance of a given structure X from the ideally
        #    interpolated counterpart X_GCH alond the i-th KPCA component (i>n)
        sigma = np.sqrt ( np.sum(np.square(l_r_sigma_s[rrank] * epsilon),axis=1) ) / sigma_etot * sigma_e

        # randomise nrg and kpca according to updated uncertainties
        # EAE : we only really need to update the kpca descriptors used for the GCH construction --> QUICKER


        nrg = np.random.normal(r_pfile[:,0],sigma[:])
        kpc = np.zeros((r_pfile[:,1::].shape))
        kpc = np.random.normal(kpc,1)
        kpc *= sigma_KPCA
        kpc += r_pfile[:,1::]

        # update input for GCH with randomised nrg
        tmp_pfile = np.column_stack((nrg,kpc))

        # construct new GCH for updated/randomised nrg
        v,l_r_sigma_s[rrank] = get_gch(tmp_pfile,inrg,cols)
        vertex_scores[v] += 1
        #vertex_list[v,n] = 1

        # evaluate probabilities r_vertex_prob based on r_vertex_scores/n
        #r_vertex_prob = r_vertex_scores*1./n
        vertex_prob = vertex_scores*1./Npp

        #if ( (n+1)%200 == 0 ) :
        #    print "Iteration : ",n+1," in ",N

    return vertex_prob

def prune_GCH(pfile,sigma_ev,convthresh,refids,nshaken,wdir,inrg=0,cols=[0,1],minprob=0.5,restart=False):

    origids = np.array(range(len(pfile)))
    # INITIAL REDUCTION OF DATASET

    # prepare reduction of dataset by constructing reference GCH
    t0=time.time()
    v,contour,sigma_s = get_refgch(pfile,inrg,cols)
    t1=time.time()
    print 'GCH construction : ',t1-t0,' sec'

    # reduce data by thresholding stabilites according to max(sigma)
    m = erfinv(1.-convthresh)*np.sqrt(2.)
    sigma = np.linalg.norm(sigma_s,axis=0)*sigma_ev # baseline energy uncertainty to be
    # used in dataset reduction before GCH sampling
    if restart==False:
        r_sigma_e = np.zeros((len( np.where(contour < m*max(sigma))[0] ))) + sigma_ev
        r_sigma_s = sigma_s[np.where(contour < m*max(sigma))[0]]
        r_pfile = pfile[np.where(contour < m*max(sigma))[0]]
        origids = origids[np.where(contour < m*max(sigma))[0]]
    else:
        r_sigma_e = np.zeros((len(sigma))) + sigma_ev
        r_sigma_s = np.zeros((sigma_s.shape)) + 0.1
        r_pfile = pfile.copy()

    # BUILDING ONE MORE REFGCH JUST TO GIVE AN IDEA
    # OF TIME PER CONVEX HULL AFTER SCREENING 

    t0 = time.time()
    get_refgch(r_pfile,inrg,cols)
    t1 = time.time()
    print "Single Hull construction during before pruning : ", t1-t0, " sec"
    ##


    # estimate uncertainty in KPCA descriptors for shaking in fuzzy GCH
    # sigmaKPCA = np.zeros((32)) + 0.01
    sigma_KPCA = eval_sampled_sigmaKPCA(refids,nshaken,wdir)

    # evaluate the energy response epsilon to changes in the KPCA
    # descriptors: dE = epsilon_i phi_i
    ndim = len(cols) # number of KPCA descriptors on which the GCH is built
    epsilon = estimate_residual_sigmaE(pfile)[ndim-1::]

    sigma_etot = np.std(pfile[:,0])

    # SAMPLE GCH
    r_vprob,r_vlist = sample_GCH(r_pfile,sigma_ev,sigma_etot,epsilon,r_sigma_s,sigma_KPCA,convthresh,refids,nshaken,wdir,inrg,cols)

    ## initialize vertex probabilities for full dataset at zero
    f_vprob = np.zeros((len(pfile)),dtype='float')
    ## translate ids in reduced dataset to full dataset and
    ## update vertex scores for full dataset at zero with entries from vertex scores for reduced dataset
    if restart==False:
        f_vprob[np.where(contour < m*max(sigma))] = r_vprob
    else:
        f_vprob = r_vprob.copy()

    vprobprune = []
    vprobprune.append(f_vprob)


    rr_vprob = r_vprob
    rr_vlist = r_vlist
    rr_pfile = r_pfile
    rr_sigma_s = r_sigma_s
    nprune = 0
    # LOOP
    #for nprune in xrange(Nprune):
    mp = 0.0
    print(" Let's start pruning! ")
    while mp < minprob:

        # REDUCTION OF DATASET
        # sort ids according to their vprob
        vids_sorted = np.argsort(rr_vprob)
        vprob_sorted = rr_vprob[vids_sorted]

        # number of struct to be pruned
        vprob_cumul = np.cumsum(vprob_sorted)
        rr_n = len(np.where(vprob_cumul < 1.0)[0])

        vids_remain = vids_sorted[rr_n::]
        rr_pfile = rr_pfile[vids_remain]
        rr_sigma_s = rr_sigma_s[vids_remain]
        rr_vids = vids_remain

        print "printing rr_pfile.shape"
        print rr_pfile.shape

        origids = origids[rr_vids]

        # SAMPLE GCH
        rr_vprob,rr_vlist = sample_GCH(rr_pfile,sigma_ev,sigma_etot,epsilon,rr_sigma_s,sigma_KPCA,convthresh,refids,nshaken,wdir,inrg,cols)

        ## initialize vertex probabilities for full dataset at zero
        f_vprob = np.zeros((len(pfile)),dtype='float')
        f_vprob[origids] = rr_vprob

        vprobprune.append(f_vprob)

        mp = min(f_vprob[f_vprob>0.0])
        print mp
        print "Pruning iter : ",nprune+1," min prob: ",mp," # vertex : ",len(np.where(vprobprune[-1]>0.0)[0])
        nprune +=1
    return vprobprune

def parallel_prune_GCH(pfile,refids,nshaken,wdir,Nprune,inrg=0,cols=[],nproc=1,convth=0.1):

    global l_r_sigma_s
    global r_pfile
    global epsilon
    global sigma_etot
    global sigma_KPCA
    global sigma
    global sigma_c
    global sigma_e



    origids = np.array(range(len(pfile)))

    # INITIAL REDUCTION OF DATASET

    # prepare reduction of dataset by constructing reference GCH
    t0=time.time()
    v,contour,sigma_s = get_refgch(pfile,inrg,cols)
    t1=time.time()
    print 'GCH construction : ',t1-t0,' sec'

    # reduce data by thresholding stabilites according to max(sigma)
    m = erfinv(1.-convth)*np.sqrt(2.)
    sigma = np.linalg.norm(sigma_s,axis=0)*sigma_e # baseline energy uncertainty to be
    # used in dataset reduction before GCH sampling

    r_sigma_e = np.zeros((len( np.where(contour < m*max(sigma))[0] ))) + sigma_e
    r_sigma_s = sigma_s[np.where(contour < m*max(sigma))[0]]
    r_pfile = pfile[np.where(contour < m*max(sigma))[0]]

    origids = origids[np.where(contour < m*max(sigma))[0]]

    # estimate uncertainty in KPCA descriptors for shaking in fuzzy GCH
    # sigmaKPCA = np.zeros((32)) + 0.01
    sigma_KPCA = eval_sampled_sigmaKPCA(refids,nshaken,wdir)

    # evaluate the energy response epsilon to changes in the KPCA
    # descriptors: dE = epsilon_i phi_i
    ndim = len(cols) # number of KPCA descriptors on which the GCH is built
    epsilon = estimate_residual_sigmaE(pfile)[ndim-1::]

    sigma_etot = np.std(pfile[:,0])


    # SAMPLE GCH

    # calculate number of GCHs to be sampled

    l_r_sigma_s =[r_sigma_s for i in rank]
    #l_r_vprob   = [np.zeros((len(r_pfile[:,1]))) for i in rank ]
    # Maps the sampleGCH routine to nproc processors
    pool = multiprocessing.Pool(processes=nproc)
    l_r_vprob = pool.map(parallel_sample_GCH,rank)


    ## Averages the probabilities accumulated in the nproc samples
    r_vprob = (np.sum(l_r_vprob,0))/nproc

    pool.close()
    ## initialize vertex probabilities for full dataset at zero
    f_vprob = np.zeros((len(pfile)),dtype='float')
    ## translate ids in reduced dataset to full dataset and
    ## update vertex scores for full dataset at zero with entries from vertex scores for reduced dataset
    f_vprob[np.where(contour < m*max(sigma))] = r_vprob


    vprobprune = []
    vprobprune.append(f_vprob)


    #rr_vprob = r_vprob
    #rr_vlist = r_vlist
    #rr_pfile = r_pfile
    #rr_sigma_s = r_sigma_s

    # LOOP
    for nprune in xrange(Nprune):

        # REDUCTION OF DATASET
        # sort ids according to their vprob
        vids_sorted = np.argsort(r_vprob)
        vprob_sorted = r_vprob[vids_sorted]

        # number of struct to be pruned
        vprob_cumul = np.cumsum(vprob_sorted)
        r_n = len(np.where(vprob_cumul < 1.0)[0])

        vids_remain = vids_sorted[r_n::]
        r_pfile = r_pfile[vids_remain]
        l_r_sigma_s = [l_r_sigma_s[i][vids_remain] for i in rank]
        r_vids = vids_remain


        origids = origids[r_vids]

        # SAMPLE GCH
        pool = multiprocessing.Pool(processes=nproc)
        l_r_vprob = pool.map(parallel_sample_GCH,rank)

        r_vprob = (np.sum(l_r_vprob,0))/nproc
	pool.close()

        ## initialize vertex probabilities for full dataset at zero
        f_vprob = np.zeros((len(pfile)),dtype='float')
        f_vprob[origids] = r_vprob

        vprobprune.append(f_vprob)

        print "Pruning iter : ",nprune+1," in ",Nprune
	np.savetxt('vprobprune.dat',vprobprune)
    return vprobprune
