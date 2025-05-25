# Carbon source effect over microbial community dynamics: A data mining approach

## Preprocessing

The preprocessing was done by the selection of time selecting the initial absolute abundance of the communities and their respective abundance after 24 hours of culture. In the cases where the 24 hour abundance is missed, a linear interpolation was done, using the `scipy.interpolate.make_interp_spline` was used with a `k=1` to make use of k-splines, as they are equivalent to linear interpolations.
