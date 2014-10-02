PeakUtils tutorial
==================

.. code-block:: python

    solver = pyfde.Solver(fitness, n_dim=2, n_pop=40, limits=(-5.12, 512))
    solver.cr, solver.f = 0.9, 0.45
