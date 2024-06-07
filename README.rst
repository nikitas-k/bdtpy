bdtpy
=====

Python port of the `Brain Dynamics Toolbox <https://bdtoolbox.org/>`_ (BDT) (all credit goes to them). I wrote this because if for some reason you don't like Matlab, you can use this package. Though the BDT is much nicer and has much more functionality. 

If you use these models, please cite their work

* Heitmann S & Breakspear M (2023) Brain Dynamics Toolbox (Version 2023a). Zenodo. `<https://doi.org/10.5281/zenodo.10112763>`_
* Heitmann S & Breakspear M (2023) Handbook for the Brain Dynamics Toolbox. 8th Edition: Version 2023. `<bdtoolbox.org>`_. Sydney, Australia. ISBN 978-0-6450669-3-7.
* Heitmann S, Aburn M, Breakspear M (2017) The Brain Dynamics Toolbox for Matlab. Neurocomputing. Vol 315. p82-88. doi:10.1016/j.neucom.2018.06.026

Installation
------------

.. code-block:: py

  git clone https://github.com/nikitas-k/bdtpy.git
  pip install bdtpy/
  # Note: do not omit the trailing slash from the above code otherwise 
  # you'll get an error (it'll look for it on pipy, bad. naughty interpreter)
  # better yet just click the "Copy" button at the top of this box

Features
--------
Only *some* SDEs are coded up for now

* Nonlinear (1D, 2D) oscillators
* Ornstein-Uhlenbeck independent processes
* Geometric Brownian Motion
* Kloeden-Platen 1999 SDE (Eq. 4.46)
* Plotting of timeseries (and phase portraits if >1D) *only* if run in a jupyter notebook
