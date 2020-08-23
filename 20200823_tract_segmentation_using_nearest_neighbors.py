# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 10:56:04 2020

@author: majaa
"""

# --- Import Dependecies
import pandas as pd
import numpy as np

import nibabel as nib

from joblib import Parallel, delayed

from dipy.viz import window, actor
from dipy.tracking.streamline import transform_streamlines
import vtk.util.colors as colors
from dipy.tracking import utils

from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.vox2track import streamline_mapping
from dipy.tracking.distances import bundles_distances_mam

from pykdtree.kdtree import KDTree

import os
import time

# --- Initialize paths
### Redirect to the path where the Code, Data and Output folder is in
os.chdir("G:\Thesis\Modified_Code") # Point to the main folder 
print(os.getcwd())

# --- Initialize Test, Train subjects and Tract name
test_subjects = ('100307', '111312', '161731')
train_subjects = ('100408', '103414', '105115', '106016', '110411', 
                 '117122', '127933', '128632', '136833', '192540', 
                 #'124422', '199655', '366446', '756055'
                 )
                 
tracts = ('af.left', 'af.right', 'cg.left', 'cg.right')
no_of_points = 12 # number of points for resampling
leafsize=10       #number of leaf size for kdtree

def load(filename, downsample=False, tree=False):
    #--- Load tractogram from TRK file 
    wholeTract= nib.streamlines.load(filename)  
    wholeTract = wholeTract.streamlines # The data is an 3D arraysequence of nibabel
    
    if(downsample):
        print("downsampling...")
        return  resample(wholeTract, tree)
    else:
        return wholeTract

def resample(streamlines, tree):
    """Resample streamlines using 12 points 
    """
    if (tree):
        return np.array([set_number_of_points(s, no_of_points).ravel() for s in streamlines]) 
    return np.array([set_number_of_points(s, no_of_points) for s in streamlines]) 
  
def show_tract(segmented_tract, color1, real_tract, color2, tract, out_path):
    """Visualization of the segmented tract.
    """ 
    os.makedirs(out_path, exist_ok=True)
    renderer = window.Renderer()
    
    affine=utils.affine_for_trackvis(voxel_size=np.array([1.25,1.25,1.25]))
    
    #--- Original tract
    bundle_native = transform_streamlines(real_tract, np.linalg.inv(affine))
    stream_actor1 = actor.line(bundle_native, colors=color2, opacity=1, linewidth=0.1)
    renderer.add(stream_actor1)
    window.record(renderer, out_path=f'{out_path}original_{tract}.png', size=(600, 600))
    
    #--- Segmented tract
    bundle_estimated = transform_streamlines(segmented_tract, np.linalg.inv(affine))
    stream_actor2 = actor.line(bundle_estimated, colors=color1, linewidth=0.1)
    #bar = actor.scalar_bar()
    renderer.add(stream_actor2)
    #renderer.add(bar)
    #window.show(renderer, size=(600, 600), reset_camera=False)          
    """Take a snapshot of the window and save it
    """
    window.record(renderer, out_path=f'{out_path}segmented_{tract}.png', size=(600, 600))
    

def dsc(estimated_tract, true_tract, tree=None):
    """Compute the overlap between the segmented tract and ground truth tract
    """
    aff=np.array([[-1.25, 0, 0, 90],[0, 1.25, 0, -126],[0, 0, 1.25, -72],[0, 0, 0, 1]])
    voxel_list_estimated_tract = streamline_mapping(estimated_tract, affine=aff).keys()
    voxel_list_true_tract = streamline_mapping(true_tract, affine=aff).keys()
    TP = len(set(voxel_list_estimated_tract).intersection(set(voxel_list_true_tract)))
    vol_A = len(set(voxel_list_estimated_tract))
    vol_B = len(set(voxel_list_true_tract))
    DSC = 2.0 * float(TP) / float(vol_A + vol_B)
    return DSC   

def get_train_sub(tract, kd_tree, downsample):
    train_y = [] # Empty list to append all the Train Subjects
    for train_sub in train_subjects: # Append all Train Subjects
        print(f"Loading {tract} from subject {train_sub}")
        y =load(f"data/train/{train_sub}/{train_sub}_{tract}.trk", downsample, kd_tree) # Load Single tracts 
        train_y = train_y + np.array(y).tolist() # Append tracts
    
    train_y = np.array(train_y)
    return train_y    

def get_test_y(test_sub, tract, kd_tree, downsample):
    
    #--- Load the specified Tract from the Test subject
    print(f"Loading {tract} from subject {test_sub}")
    test_y =load(f"data/test/{test_sub}/{test_sub}_{tract}.trk", downsample, kd_tree)
            
    return test_y

def get_test_x(test_sub, kd_tree, downsample):
    
    #--- Load the whole brain from the Test subject
    print(f"Loading whole brain from subject {test_sub}")
    test_X = load(f"data/test/{test_sub}/full1M_{test_sub}.trk", downsample, kd_tree)       
    
    return test_X


def bundles_distances_mam_smarter_faster(A, B, n_jobs=-1, chunk_size=100):
    """Parallel version of bundles_distances_mam that also avoids
    computing distances twice.
    """
    lenA = len(A)
    chunks = chunker(A, chunk_size)
    if B is None:
        dm = np.empty((lenA, lenA), dtype=np.float32)
        dm[np.diag_indices(lenA)] = 0.0
        results = Parallel(n_jobs=-1)(delayed(bundles_distances_mam)(ss, A[i*chunk_size+1:]) for i, ss in enumerate(chunks))
        # Fill triu
        for i, res in enumerate(results):
            dm[(i*chunk_size):((i+1)*chunk_size), (i*chunk_size+1):] = res
            
        # Copy triu to trid:
        rows, cols = np.triu_indices(lenA, 1)
        dm[cols, rows] = dm[rows, cols]

    else:
        dm = np.vstack(Parallel(n_jobs=n_jobs)(delayed(bundles_distances_mam)(ss, B) for ss in chunks))

    return dm

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def build_kdtree(points):
    """Build kdtree with resample streamlines 
    """
    return KDTree(points,leafsize)    
    
def kdtree_query(tract, kd_tree):
    """compute 1 NN using kdtree query and return the id of NN
    """
    print(type(tract), tract.shape) 
    dist_kdtree, ind_kdtree = kd_tree.query(np.float32(tract), k=1)
    return np.hstack(ind_kdtree) 

def find_nearest(train_y, dist_mat):
    nearest=[0 for y in range(len(train_y))] 
    for l in range(0,len(train_y)) :     
        m,k=min((v,i) for i,v in enumerate(dist_mat[l]))
        if k not in nearest:
            nearest.append(k)
        
    return nearest
    
def segmentation_with_NN(calc_dist=None, downsample=None, tree=None):
        
    #--- Laod Train subjects
    for tract in tracts:
        train_y = get_train_sub(tract, tree, downsample=True)
        
        for test_sub in test_subjects:
            test_X = get_test_x(test_sub, tree, downsample)
            t0 = time.time()
            
            if (tree==None):
                #--- Calculate the Distance Matrix
                if (calc_dist == 'simple'):
                    dist_mat = bundles_distances_mam(train_y , test_X)
                elif (calc_dist == 'fast'):
                    print("fast...")
                    dist_mat = bundles_distances_mam_smarter_faster(train_y , test_X)
                
                #--- Find the Streamlines with minimum distance
                nearest_streamlines = find_nearest(train_y, dist_mat)
            
            elif (tree):
                #build kdtree
                print("Buildng kdtree")
                kd_tree=build_kdtree(test_X)
                
                #kdtree query to retrive the NN id
                nearest_streamlines=kdtree_query(train_y, kd_tree)
            
            print(f"Time needed = {time.time()-t0}")
            test_X = get_test_x(test_sub, False, False) # Get the original brain with all streamlines
            test_y = get_test_y(test_sub, tract, False, False) # Get the original tract with all streamlines
            
            #--- Estimated tracts using Nearest Neighbor from the Test Subject
            estimated_tract = test_X[nearest_streamlines]
            #real_tract = test_X[test_y.index]
            
            print(f"Computing Dice Similarity Coefficient of {tract} from test subject {test_sub}......")               
            print ("DSC= %f" %dsc(estimated_tract,test_y, tree))          
            
            out_path = f"output/test/{test_sub}/"      
            print(f"Showing {tract} from test subject {test_sub}......")               
            color_positive= colors.green
            color_negative=colors.red
            show_tract(estimated_tract, color_positive, test_y, color_negative, tract, out_path)
            
            
        
    
#--- MAIN ---------------------------------------  
print("Segmenting tract with NN......")   
# --- For SDMC, calc_dist='simple', downsample=None, tree=None
# --- For FDMC, calc_dist='fast', downsample=None, tree=None
# --- For FDMC+DS, calc_dist='fast', downsample=True, tree=None
# --- For KdTree, calc_dist=None, downsample=True, tree=True

segmentation_with_NN(calc_dist=None, downsample=True, tree=True)






