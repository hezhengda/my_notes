Practical use of NequIP
===========================

Install NequIP
----------------

By following the commands below, you should be able to install NequIP on your local machine or HPC (e.g. Kestrel from NREL).

.. code-block:: bash

    module load anaconda3
    conda create -n nequip python 
    conda activate nequip 
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 # for CUDA 12.4
    git clone https://github.com/mir-group/nequip.git
    cd nequip; pip3 install .

In order to test whether the installation is successful, we can run the following command:

.. code-block:: bash

    cd nequip
    nequip-train config/minimal.yaml

If the installation is successful, you should see the following output:

.. figure:: ../_static/img/nequip_success.png
    :width: 800px
    :align: center

    If you get similar outputs as above, the installation is successful. 

Apply for an interactive mode on Kestrel
----------------------------------------

.. code-block:: bash

    salloc --account=cmos --time=6:00:00 --gres=gpu:1 --mem-per-gpu=80G

Generating training data for NequIP
--------------------------------------
Here's the script (:code:`extract_ase_split.py`) for generating training data for NequIP.

It's usage is:

:: 

    python extract_ase_split.py -n 10 --n_jobs 10000 --conv_jobs_file convJobs_Relax.mson --split 0.8 0.1 0.1
    # -n 10: use 10 cores for parallel processing
    # --n_jobs 10000: process 10000 jobs
    # --conv_jobs_file convJobs_Relax.mson: the file that contains the location of converged jobs
    # --split 0.8 0.1 0.1: split the data into 80% training, 10% validation, 10% test

Here's the script for generating training data for NequIP.

.. code-block:: python

    import os
    import argparse
    from random import shuffle
    from multiprocessing import Pool
    from monty.serialization import loadfn, dumpfn
    from ase.io import read, write
    import numpy as np

    def process_job(job):
        print(f"Processing {job}")
        os.chdir(job)

        if 'vasprun.xml.xz' in os.listdir():
            os.system('xz --decompress vasprun.xml.xz')
        lst_structures = read('vasprun.xml', index=':') # index=':' to read all structures
        os.system('xz vasprun.xml')

        # Extract unique elements from the atoms object
        unique_elements = list(set(lst_structures[-1].get_chemical_symbols()))

        return lst_structures, unique_elements

    def collect_unique_elements(results):
        all_elements = set()
        for _, elements in results:
            all_elements.update(elements)
        return sorted(list(all_elements))

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Extract ASE structures in parallel.")
        parser.add_argument("-n", "--num_cores", type=int, default=1, help="Number of cores to use")
        parser.add_argument("--n_jobs", type=int, default=None, help="Number of jobs to process")
        parser.add_argument("--conv_jobs_file", type=str, default="convJobs_Relax.mson", help="Filename of conv_jobs")
        parser.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1], 
                            help="Split ratio for training, validation, and test sets")
        args = parser.parse_args()

        # Ensure split ratios sum to 1
        if sum(args.split) != 1:
            raise ValueError("Split ratios must sum to 1")

        currloc = os.getcwd()
        ase_structures_dir = f'{currloc}/ase_structures'
        os.makedirs(ase_structures_dir, exist_ok=True)

        all_conv_jobs = loadfn(f'{currloc}/{args.conv_jobs_file}')
        shuffle(all_conv_jobs) # shuffle the jobs to avoid bias
        n_jobs = min(args.n_jobs, len(all_conv_jobs)) if args.n_jobs else len(all_conv_jobs)
        conv_jobs = all_conv_jobs[:n_jobs]

        with Pool(processes=args.num_cores) as pool:
            results = pool.map(process_job, conv_jobs)

        # Collect all structures
        all_structures = [struct for result in results for struct in result[0] if result[0] is not None]

        filter_structures = []
        for struct in all_structures:
            try:
                forces = struct.get_forces()
            except:
                all_structures.pop(struct)
        
        # Shuffle all structures
        shuffle(all_structures)

        # Split structures based on the provided ratios
        total_structures = len(all_structures)
        split_indices = [int(ratio * total_structures) for ratio in args.split]
        
        train_structures = all_structures[:split_indices[0]]
        val_structures = all_structures[split_indices[0]:split_indices[0]+split_indices[1]]
        test_structures = all_structures[split_indices[0]+split_indices[1]:]

        # Collect unique elements
        unique_elements = collect_unique_elements(results)

        # Write structures to files
        write(f'{ase_structures_dir}/training.xyz', train_structures, format='extxyz')
        write(f'{ase_structures_dir}/validation.xyz', val_structures, format='extxyz')
        write(f'{ase_structures_dir}/test.xyz', test_structures, format='extxyz')
        
        # Save unique elements to a .mson file
        dumpfn(unique_elements, f'{ase_structures_dir}/unique_elements.mson')

        print(f"Unique elements: {unique_elements}")
        print(f"Processed {len(train_structures)} training structures")
        print(f"Processed {len(val_structures)} validation structures")
        print(f"Processed {len(test_structures)} test structures")

After the splitting of the DFT results, you will find :code:`training.xyz`, :code:`validation.xyz`, :code:`test.xyz` in the directory :code:`ase_structures/`. You can easily change the location by adding additional arguments to the script.

Training a NequIP model
--------------------------------

Here's the script for training a NequIP model.

.. code-block:: bash

    nequip-train config/your_config.yaml

After successful training, you will get :code:`best_model.pth` in the directory :code:`results/`.

Evaluate the trained model
--------------------------------

Here's the script for evaluating the accuracy of predicting total energy for a trained NequIP model.

.. code-block:: python

    from tqdm import tqdm
    import numpy as np
    from ase.io import read
    from nequip.ase import NequIPCalculator

    lst_atoms = read('test.xyz', index=':')

    nequip_calc = NequIPCalculator.from_deployed_model(model_path='/home/hez/nequip/results/IM_surf/IM_surf/deployed_model.pth')

    lst_e_dft = []
    lst_e_nequip = []

    for atoms in tqdm(lst_atoms):
        lst_e_dft.append(atoms.get_potential_energy())

        # reset the calculation
        atoms.calc = nequip_calc
        lst_e_nequip.append(atoms.get_potential_energy())

    np_e_dft = np.array(lst_e_dft)
    np_e_nequip = np.array(lst_e_nequip)

    print(np.corrcoef(np_e_dft, np_e_nequip)) 

If you want to use NequIP as a force field, you can link it with LAMMPS code.