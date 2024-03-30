import FiberTracker as ft

def fibertracker(int_thres=0.1, max_jump=5, min_dist=5, max_skip=5, peak_sigma=2, center_sigma=1.5, momentum=0.1, track_min_length=3):
    return ft.FiberTracker(int_thres=int_thres, max_jump=max_jump, min_dist=min_dist, max_skip=max_skip, 
                           peak_sigma=peak_sigma, center_sigma=center_sigma, momentum=momentum, 
                           track_min_length=track_min_length)
