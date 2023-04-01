"""
Created on Sat Apr  1, 2023

@author: Anders Bjorholm Dahl
e-mail: abda@dtu.dk

This program is free software: you can redistribute it and/or modify it under the 
terms of the GNU General Public License as published by the Free Software Foundation, 
either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY 
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program. If not, see <https://www.gnu.org/licenses/>.
"""

# Class for fiber tracking


import tifffile
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import os

class FiberTracker:
    def __init__(self, int_thres=0.1, d_thres=5, count_thres=5, peak_sigma=2, center_sigma=1.5, momentum=0.1, length_thres=3):
        '''
        Initialization of the fiber tracker

        Parameters
        ----------
        int_thres : float, optional
            Threshold for minimum value of areas containing fibers. The default is 0.1.
        d_thres : float, optional
            Distance threshold in pixels. The default is 5.
        count_thres : int, optional
            Minimum number of points in one fiber. The default is 5.
        peak_sigma : float, optional
            Parameter for Gaussian smoothing the input image before finding peaks. 
            The default is 2. If negative, the no smoothing.
        center_sigma : float, optional
            Parameter for Gaussian smoothing center detection points for creating 
            a volume of center lines. The default is 1.5. If negative, the no smoothing.
        momentum : float, optional
            Parameter in the range [0;1] that gives momentum to the tracking direction. 
            The momentum gives the fraction of the current direction that is kept.
            The default is 0.1.
        length_thres : int, optional
            Minimum number of points in a track.

        Returns
        -------
        None.

        '''
        self.int_thres = int_thres
        self.d_thres = d_thres**2 # not move more than d_thres pixels (using squared distance)
        self.count_thres = count_thres # minimum number of points in a track
        self.peak_sigma = peak_sigma
        self.center_sigma = center_sigma
        self.momentum = momentum
        self.length_thres = length_thres

        
    def __call__(self, V):
        '''
        Call function for FiberTracker

        Parameters
        ----------
        V : numpy array
            3D array of fibers aligned with the first dimension.

        Returns
        -------
        list
            list of numpy arrays each containing coordinates of tracked fibers.
        numpy array
            Volume of center points smoothed with center_sigma. Useful for 3D volume rendering.

        '''
        if V.ndim != 3:
            print(f'Input volume is {V.ndim}, but should be 3-dimensional!')
            return 0, 0
        else:
            V = V.astype(float)
            V /= V.max()
            r, c, Cv = self.find_peaks_vol(V)
            return self.track_fibers(r, c, V), Cv
        
    def find_peaks(self, im):
        '''

        Parameters
        ----------
        im : numpy array
            Image.

        Returns
        -------
        r : numpy array
            1D array of row coordinates of peaks.
        c : numpy array
            1D array of column coordinates of peaks.
        cim : numpy array
            2D image of center points

        '''        
        ctim = np.zeros(im.shape)
        
        disp = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]
        for d in disp:
            ctim[1:-2,1:-2] += im[1:-2,1:-2] > im[1+d[0]:-2+d[0],1+d[1]:-2+d[1]]

        cim = ( ctim==8 ) & ( im > self.int_thres )
        r, c = np.where(cim)
        return r, c, cim
    
    def get_dist(self, ra, ca, rb, cb):
        '''
        Parameters
        ----------
        ra : numpy array
            1D array of row coordinates of point a.
        ca : numpy array
            1D array of column coordinates of point a.
        rb : numpy array
            1D array of row coordinates of point b.
        cb : numpy array
            1D array of column coordinates of point b.

        Returns
        -------
        numpy array
            n_a x n_b 2D eucledian distance array between the two point sets.

        '''
        ra = np.array(ra)
        ca = np.array(ca)
        rb = np.array(rb)
        cb = np.array(cb)
        return ((ra.reshape(-1,1)@np.ones((1,len(rb))) - np.ones((len(ra),1))@(rb.reshape(1,-1)))**2 + 
                (ca.reshape(-1,1)@np.ones((1,len(cb))) - np.ones((len(ca),1))@(cb.reshape(1,-1)))**2)


    def remove_close_points(self, r, c, im):
        '''

        Parameters
        ----------
        r : numpy array
            1D array of row coordinates.
        c : numpy array
            1D array of column coordinates.
        im : numpy array
            2D image with peak intensities.

        Returns
        -------
        numpy array
            1D array of kept row coordinates.
        numpy array
            1D array of kept column coordinates.

        '''
        d = self.get_dist(r, c, r, c)
        d += np.eye(len(d))*1e10
            
        keep = np.ones(len(d), dtype=bool)
        
        id_thres = np.where(d<self.d_thres)
        
        for ida, idb in zip(id_thres[0], id_thres[1]):
            if im[r[ida], c[ida]] > im[r[idb], c[idb]]:
                keep[idb] = 0
        
        return r[keep], c[keep]
    
    
    def swap_place(self, tr, id_first, id_second):
        '''

        Parameters
        ----------
        tr : list
            List of elements.
        id_to : integer
            Index of first element.
        id_from : integer
            Index of second element.

        Returns
        -------
        tr : list
            Updated list of elements.

        '''
        tmp = tr[id_first]
        tr[id_first] = tr[id_second]
        tr[id_second] = tmp
        return tr
    
    def find_peaks_vol(self, V):
        '''
        Find peaks in the volume and set the values of r (peak rows coordinates), 
        c (peak column coordinates), and Cv (volume of center coordiantes smoothed 
        by cetner_sigma)

        Returns
        -------
        None.

        '''
        r = []
        c = []
        cv = []

        for z, v in enumerate(V):
            if self.peak_sigma > 0:
                v = scipy.ndimage.gaussian_filter(v, self.peak_sigma)
            r_out, c_out, cim = self.find_peaks(v)
            r.append(r_out)
            c.append(c_out)
            if self.center_sigma > 0:
                cim = scipy.ndimage.gaussian_filter(cim.astype(float), self.center_sigma)
            cv.append(cim)
        Cv = np.stack(cv)
        Cv = (Cv/Cv.max()*255).astype(np.uint8)
        
        return r, c, Cv




    def track_fibers(self, r, c, V):
        '''
        Tracks fibers throughout the volume. Sets the parameters 
            - rr and cc (row and column coordinates where poitns closer than 
              d_thres have been removed. The point with highes intensity is kept)
            - tracks_all, which is a lsit of all tracked fibers
            - tracks, which is a list of tracks that are longer than count_thres

        Returns
        -------
        None.

        '''
        
        tracks_all = [] # Coordinates
        ntr_ct = [] # count of not found points
        
        # remove close points
        rr = []
        cc = []
        for ra, ca, v in zip(r, c, V):
            rra, cca = self.remove_close_points(ra, ca, v)
            rr.append(rra)
            cc.append(cca)
            
        # initialize tracks (row, col, layer, drow, dcol) and counter for tracks
        for ra, ca in zip(rr[0], cc[0]):
            tracks_all.append([(ra, ca, 0, 0, 0)])
            ntr_ct.append(0)

        coord_r = rr[0].copy()
        coord_c = cc[0].copy()
        
        nf_counter = 0
        for i in range(1, len(rr)):
            
            # Match nearest point
            d = self.get_dist(coord_r, coord_c, rr[i], cc[i])
            
            id_from = d.argmin(axis=0) # id from
            id_to = d.argmin(axis=1) # id to
            
            d_from = d.min(axis=0)
                
            id_match_from = id_to[id_from] # matched id from
            idx = id_match_from == np.arange(len(id_from)) # look up coordinates
            for j in range(len(idx)):
                if idx[j] and d_from[j] < self.d_thres:
                    drow = (self.momentum*(rr[i][j] - tracks_all[id_from[j] + nf_counter][-1][0]) + 
                            (1-self.momentum)*tracks_all[id_from[j] + nf_counter][-1][3])
                    dcol = (self.momentum*(cc[i][j] - tracks_all[id_from[j] + nf_counter][-1][1]) +
                            (1-self.momentum)*tracks_all[id_from[j] + nf_counter][-1][4])
                    tracks_all[id_from[j] + nf_counter].append((rr[i][j], cc[i][j], i, drow, dcol))
                else:
                    tracks_all.append([(rr[i][j], cc[i][j], i, 0, 0)])
                    ntr_ct.append(0)
                    
            not_matched = np.ones(len(coord_r), dtype=int)
            not_matched[id_from] = 0
            for j in range(len(not_matched)):
                if not_matched[j]:
                    ntr_ct[j + nf_counter] += 1
            
            coord_r = []
            coord_c = []
                        
            for j in range(nf_counter, len(tracks_all)):
                if ntr_ct[j] > self.count_thres:
                    ntr_ct = self.swap_place(ntr_ct, j, nf_counter)
                    tracks_all = self.swap_place(tracks_all, j, nf_counter)
                    nf_counter += 1
            
            for j in range(nf_counter, len(tracks_all)):
                coord_r.append(tracks_all[j][-1][-5] + (i-tracks_all[j][-1][-3])*tracks_all[j][-1][-2])
                coord_c.append(tracks_all[j][-1][-4] + (i-tracks_all[j][-1][-3])*tracks_all[j][-1][-1])
        
        tracks = []
        for track in tracks_all:
            if len(track) > self.length_thres:
                track_arr = np.stack(track)
                tracks.append(track_arr[:,:3])
        
        return tracks


if __name__ == '__main__':
    def add_slash(name):
        if not name[-1]=='/':
            name += '/'
        return name
    
    dir_name = 'data/'
    dir_name = add_slash(dir_name)
    V = tifffile.imread(f'{dir_name}vol_center_probability.tif')
    fib_tracker = FiberTracker()
    tracks, Cv = fib_tracker(V)
    
    out_dir = 'results/'
    out_dir = add_slash(out_dir)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    tifffile.imwrite(f'{out_dir}center_vol.tif', Cv)
    
    n_rand = 100
    rid = np.random.choice(len(tracks), n_rand)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in rid:
        ax.plot(tracks[i][:,0], tracks[i][:,1], tracks[i][:,2], '-')
    ax.set_aspect('equal')



