Aqueous stability of metal and metal (oxy)(hydr)oxides
=========================================================

.. note::

    The purpose of this project is to check the aqueous stability of metal and metal (oxy)(hydr)oxides and develop some statistical insights of the results.

Abstract
--------

Introduction
------------

Methods
-------

Collect a dataset of metal and metal (oxy)(hydr)oxide compounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have extract all the compositions and corresponding Pourbaix diagrams from Materials Project. The MP interface API was used to extract all the data. The data extraction script for all compositions (:code:`extract_comps.py`) is shown in below:

.. dropdown:: extract_comps.py
    :animate: fade-in

    .. include:: aqueous_stability/extract_comps.py
        :code: python

Some notes about the script above:

- We are using `multiprocessing` to speed up the data extraction.
- We are using `pickle` to save and re-read the data more efficiently.
- The :code:`aqueous_stability` package is a custom package for this project, which we will put in the appendix in this page.

The actual data extraction was done on :code:`Kestrel`. The folder is: :code:`/home/hez/new_pourbaix_comp`. Besides the extraction code, I also write a shell script to run it automatically and dealing with issues like the server blocks me because of too many requests.

.. dropdown:: extract_comps.sh
    :animate: fade-in

    .. code-block:: bash
        :linenos:

        #!/usr/bin/sh
        for i in {1..10000}
        do
            echo "$i-th loop" >> log_auto
            python3 extract_comps.py -n 24 -c 3
            echo "sleep for 2 minutes to make sure the server forgets about me ......"
            sleep 2m
            echo "come back to life and go attack again"
            echo >> log_auto
        done

The results were saved in :code:`/Users/zhengdahe/Library/CloudStorage/Dropbox/proj_StabilityWindow_Catalysts/figures_addInGePbLaBi/Figure_1`.

Collect pourbaix diagrams
~~~~~~~~~~~~~~~~~~~~~~~~~~

Besides all the compositions mentioned above, we also collected all relevant pourbaix diagrams constructed by using :code:`pymatgen` package. The code for extracting pourbaix diagrams (:code:`extract_pbds.py`) is shown in below:

.. dropdown:: extract_pbds.py
    :animate: fade-in

    .. include:: aqueous_stability/extract_pbds.py
        :code: python

The executing script (:code:`extract_pbds.sh`) is similar to the one for compositions.

.. important::

    - When we generate the pourbaix diagram, the composition should be the same as the target compound. For example, if we want to generate the pourbaix diagram for :math:`\rm Li_2MoO_4`, then the composition of the Pourbaix diagram should be :code:`{'Li':2,'Mo':1}`. An advantage of this is that we don't have solid compounds that is stable in multiple regions. 

    - After we have generated the Pourbaix diagram, we need to screen again to find the missing stable solid entries. Then we will search for the new Pourbaix diagram until all the stable solid entries are found, which means no new solid entries were found.

.. dropdown:: extract_pbds.sh
    :animate: fade-in

    .. code-block:: bash
        :linenos:

        #!/usr/bin/sh
        for i in {1..10000}
        do
            echo "$i-th loop" >> log_auto
            python3 extract_pbds.py -n 12 -c 3
            echo "sleep for 2 minutes to make sure the server forgets about me ......"
            sleep 2m
            echo "come back to life and go attack again"
            echo >> log_auto
        done
        
Determine the stability range of pH and voltage for each compound
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to get the stability range of pH and voltage for each compound, we need to do the Pourbaix diagram analysis. First, we need to determine the stable region for each compound. Then we need to find the bounary based on the vertices of each region. In below is a schematic diagram for this process:

.. figure:: ../_static/img/stable_pH_V_range.png
    :width: 600px
    :align: center

    Schematic diagram for determining the stable pH and V range for a given compound.

There are two steps to get the stable pH and V range for a given compound:

1. Determine the stable region(s) for each compound. 
2. Find the bounary based on the combined vertices of all regions.

Let's look at the first point, the code is in :code:`aqueous_stability.py`. I only show the most relevant part in below.

.. admonition:: Algorithm

    - **Step 1**: Scan all the stable entries in the pourbaix diagram (:code:`pbd.get_pourbaix_domains`)
    - **Step 2.1**: For each stable entry, if it is :code:`PourbaixEntry` (with :code:`phase_type` attribute), then check if it is solid. If it is not solid, then skip it. If it is solid, then check if it is the target phase. If it is, then add it to the list.
    - **Step 2.2**: For each stable entry, if it is not :code:`PourbaixEntry` (i.e. it is a :code:`PourbaixMultiEntry`), then for each entry in the :code:`entry_list`, check if it is solid. If it is not solid, then skip it. If it is solid, then check if it is the target phase. If it is, then add it to the list.
    - **Step 3**: Return the list of all stable entries that are solid and the target phase.

Once we have the stable solid entries, we can determine its stable region, and figure out the shared region with stable region of water (:math:`\rm pH=0\rightarrow 14, U_{RHE}=0\rightarrow 1.23V`)

Results 
--------

Statistical analysis
~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../_static/img/aqueous_stability_compounds_statistics.png
    :width: 600px
    :align: center

    Statistical analysis of all stable solid compounds. (a) Selected elements (colored in dark blue) shown in the periodic table. (b) Number of compounds for each material family. (c) The distribution of the number of compounds for each family across different elements

Distribution of stable regions 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../_static/img/aqueous_stability_stable_region_distribution.png
    :width: 600px
    :align: center

    Distribution of stable regions for all families of compounds: (a) metal (b) intermetllic compounds (c) binary oxides (d) ternary and higher-order oxides. For each family, three examples were given, the corresponding color
    
Stability of binary oxides
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../_static/img/aqueous_stability_stability_binary_oxides.png
    :width: 600px
    :align: center

    Stable (a) pH and (b) voltage ranges for binary oxides. In (a), the dashed grey line represents :math:`\rm pH=7`. The oxidation state of the metal atoms are shown in different saturation, with the highest oxidation state shown in the darkest color, the corresponding colorbar is shown on the right. The value of corresponding oxidation state for each bar is also shown on both panels.

Stability of ternary and higher-order oxides 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

.. figure:: ../_static/img/aqueous_stability_stable_ternary_higher_oxides.png
    :width: 600px
    :align: center

    Stability of ternary and higher-order oxides with comparison to decomposed binary oxides. (a) All decomposed binary oxides are stable and the stable regions overlap. The stable pH and voltage ranges for two selected compounds are shown in (d) and (g), respectively. The black vertial line divides two examples. (b) All decomposed binary oxides are stable, but they do not overlap. The stable pH and voltage ranges for two selected compounds are shown in (e) and (h), respectively. (c) Not all decomposed binary oxides are stable. The stable pH and voltage ranges for two selected compounds are shown in (f) and (i), respectively. 
    

Discussion
-----------

Synergistic and Antagonistic effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this part, we will explore the synergistic and antagonistic effects of intermetallic compounds and oxides. The definition of the synergistic and antagonistic effects are:

- Synergistic effects: 
    - condition 1: (element A not stable) or (element B not stable)
    - condition 2: (element A is stable) and (element B is stable) and (element A & B do not overlap) and (AB is stable)
- Antagonistic effects:
    - (element A is stable) and (element B is stable) and (AB is not stable)

For intermetallic compounds, the A and B are pure metal phases, but for oxides, A and B represents the binary oxide phases.

Synergistic effects
^^^^^^^^^^^^^^^^^^^^

.. figure:: ../_static/img/aqueous_stability_syn_im_ox.png
    :width: 600px
    :align: center

    Heatmap of synergistic effects for intermetallic compounds (lower left) and oxides (upper right). Each block represents the number of compounds that exhibit synergistic effects.

Antagonistic effects
^^^^^^^^^^^^^^^^^^^^

.. figure:: ../_static/img/aqueous_stability_ant_im_ox.png
    :width: 600px
    :align: center

    Heatmap of antagonistic effects for intermetallic compounds (lower left) and oxides (upper right). Each block represents the number of compounds that exhibit antagonistic effects.


Conclusion
----------

References
----------

Appendix
--------

This is all the code for :code:`aqueous_stability.py` file. It defines the :code:`AqueousStabilityWorker` class, which is used to handle the pourbaix diagram and also the stability analysis of compounds.

.. dropdown:: aqueous_stability.py
    :animate: fade-in

    .. include:: aqueous_stability/aqueous_stability.py
        :code: python 

This is the code for :code:`util_aqueous_stability.py` file, which contains some utility functions for the aqueous stability analysis. Also all the figures can be generated by using the functions in this file.

.. dropdown:: util_aqueous_stability.py
    :animate: fade-in

    .. include:: aqueous_stability/util_aqueous_stability.py
        :code: python