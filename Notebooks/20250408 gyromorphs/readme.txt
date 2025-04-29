20250417_gyromorphs_2D_data contains three keys: 

transmission_data: array with shape (3,400) the three elements correspond in this order to the results for (gyro24,gyro60,shu_chi_0.4), and 400 frequencies
nu: reduced frequencies a/lambda 
frequencies: raw frequencies in Hz 

The gyro samples were scaled up so, the characteristic length a \approx 1 as in the SHU experiments. 

We used disk of perm 11.56 and radius 0.189a as in L. S. Froufe-Perez,et al, PNAS 114, 9570 (2017).

We took slabs of size 25x25a for this 

We used periodic boundary conditions perpendicular to the direction of propagation, and absorbers at the ends of the simulation cell. 

The source is a TM polarized planewave. 

