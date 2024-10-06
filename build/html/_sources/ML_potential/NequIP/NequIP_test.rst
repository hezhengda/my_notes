Test of NequIP
===============

In this section, we will do a comprehensive test of NequIP. 

The dataset we used is from our in-house high-entropy alloy (HEA) surfaces dataset. In total it contains around 12,000 HEA surfaces. 

We are going to test on two supercomputers: (1) Supercomputer :code:`Kestrel` at NREL, (2) Supercomputer :code:`Perlmutter` at NERSC. I will summarize all the tests, the codes and the results that I have done in here. Hope this can be a good reference for future work.

Test 1: Scaling tests on both Perlmutter and Kestrel
----------------------------------------------------

.. admonition:: Plans

    Do the scaling tests on both Perlmutter and Kestrel using dataset contains 1024 samples. Tests will be done on: (a) 1 node, using 1/2/4 GPUs (b) use 1/2/4 nodes and all GPUs available. For each job, run 10 epochs.

I need to write a script to extract the :code:`total walltime` from the output folders. I can get the run_folder from the :code:`run.yaml` file using :code:`awk '{print $2}'`. Then I can read the output csv files.

Test 2: Learning curves using parameters provided by the package
-----------------------------------------------------------------

.. admonition:: Plans

    Get the learning curves using the basic parameters provided by the package. The size of dataset is: 512, 1024, 2048, 4096, 8192.

Test 3: Tuning the parameters to get the best performance
-----------------------------------------------------------

.. admonition:: Plans

    Here are the parameters that we want to investigate: 
