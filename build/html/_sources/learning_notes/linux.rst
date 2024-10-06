Useful commands on Linux
==========================

rsync
-----

:code:`rsync` is a command-line utility for efficiently transferring and synchronizing files between hosts over a network. It is designed to be fast, secure, and reliable.

.. code-block:: bash

    # Kestrel
    rsync hez@kestrel.nrel.gov:/home/hez/nequip_multiGPU/IM_surf_prod/tests/scaling_test/dict_analysis.mson .

    # Perlmutter
    rsync zhe1@perlmutter-p1.nersc.gov:/pscratch/sd/z/zhe1/IM_surf_prod_NequIP/tests/scaling_test/dict_analysis.mson .

awk 
-----

.. code-block:: bash

    awk '{print $2}' file.txt # print the second column of the file

transfer files between two places 
----------------------------------

A trick to do it effectively is to use environment variable to specify the location of the cluster, for example, the perlmutter cluster can be written as: `zhe1@perlmutter-p1.nersc.gov`, I can assign an environment variable as $PERLMUTTER_PATH:

.. code-block:: bash

    export PERLMUTTER_PATH=zhe1@perlmutter-p1.nersc.gov

So next time I only need to use `$PERLMUTTER_PATH` to specify the prefix of the cluster.