Theoretical foundation of M3GNET
=================================

`M3GNET <https://www.nature.com/articles/s43588-022-00349-3>`_ is a graph neural network used for training interatomic potential based on DFT data. In this page, we will introduce its network architecture and the detailed mathematics used in the network.

Network architecture
--------------------

The architecture of the network is shown in :numref:`fig-m3gnet`.

.. figure:: ../../_static/img/m3gnet.png
   :width: 300px
   :align: center
   :name: fig-m3gnet
   :alt: M3GNET architecture

   M3GNET architecture. Copied from the original paper (Fig. 1).

The basic architecture is typical graph neural network: featurization (structure to graph) :math:`\longrightarrow` message passing (updating / convolution) :math:`\longrightarrow` global pooling :math:`\longrightarrow` output.

Molecular graph construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First we will featurize the atom (as nodes or vertices) and the bonds between atoms (as edges) into a graph. 

In M3GNET, a graph object can be written as: :math:`\mathcal{G} = (\mathcal{E},\mathcal{V},\mathcal{X}[\textbf{M}, \textbf{u}])`, where :math:`\mathcal{E}` represents the collection of edges (:math:`\{\vec{e}_{ij}\}`), :math:`\mathcal{V}` represents the collection of vertices (:math:`\{\vec{v}_i\}`), :math:`\mathcal{X}` represents the location of all atoms, :math:`\textbf{M}` represents the lattice constant of the material, and :math:`\textbf{u}` represents the global feature vector, which can be used in enconding some additional information about the material, or the computational approaches. For example, in this `work <https://www.nature.com/articles/s43588-020-00002-x>`_, they used global feature vector to encode the fidelity of the DFT calculation. Here are some further development in this direction: `paper <https://arxiv.org/abs/2409.00957>`_.

In :code:`Graph featurizer`, the nodes (:math:`\vec{v}_i`) are featurized into a :math:`\vec{v}_i \in \mathbb{R}^{64\times 1}` vector. The edges (:math:`\vec{e}_{ij}`) are featurized into a :math:`\vec{e}_{ij} \in \mathbb{R}^{m\times 1}` vector, where :math:`m` is the number of basis sets. The basis functions (:math:`h_m(r_{ij})`) are shown in :numref:`fig-basis-functions`.

.. figure:: ../../_static/img/basis_functions_m3gnet_edge.png
   :width: 400px
   :align: center
   :alt: Basis functions
   :name: fig-basis-functions

   Basis functions. Copied from the original paper (in Methods section).

In the following, :math:`e_{ij}^0` is used as a vector with length :math:`m`, it can be written as:

.. math::

   e_{ij}^0 = \begin{bmatrix}
   h_1(r_{ij}) \\
   h_2(r_{ij}) \\
   \vdots \\
   h_m(r_{ij}) \\
   \end{bmatrix}

Message passing
~~~~~~~~~~~~~~~

After the featurization of the nodes and edges, we can update the nodes and edges using the message passing approach. There are three parts needed to be updated: (1) update the nodes using the edges, (2) update the edges using the nodes, (3) update the global feature vector using the nodes. We will introduce them individually.

Step 1: Edge feature update 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In M3GNET, the update of edge features needs two steps: the first step is to get the many-body interaction, the second step is like the regular updating of edge features.

Many-body Interaction
**********************

For the many-body interaction, a Tersoff-like potential is used:

.. math::

    (\tilde{e}_{ij})_{n,l} = \sum_k j_l(z_{ln}\frac{r_{ik}}{r_c})Y_l^0(\theta_{ijk})\odot \sigma(W_v\vec{v}_k+b_v)f_c(r_{ij})f_c(r_{ik})

where :math:`j_l(z_{ln}\frac{r_{ik}}{r_c})` is the spherical Bessel function, :math:`Y_l^0(\theta_{ijk})` is the spherical harmonic function, with :math:`\theta_{ijk}` as the angle between the vectors :math:`\vec{e}_{ij}` and :math:`\vec{e}_{ik}`, which can be considered as the three-body interaction, that's why this model is called :code:`M3GNET`, generally it should be called :code:`MnGNET`, where :math:`n` is the number of body interactions considered, in the current case :math:`n=3`. 

:math:`\sigma(W_v\vec{v}_k+b_v)` is the activation function, :math:`f_c(\cdot)` is the cutoff function, the expression is: :math:`f_c(r_{ij}) = 1-6(r_{ij}/r_c)^5+15(r_{ij}/r_c)^4-10(r_{ij}/r_c)^3`. :math:`\odot` represents the element-wise multiplication. 

Then the edge features can be updated as:

.. math::

    \vec{e}_{ij}^{'} = \vec{e}_{ij} + g(\tilde{W}_2\tilde{e}_{ij}+\tilde{b}_2)\odot \sigma(\tilde{W}_1\tilde{e}_{ij}+\tilde{b}_1)

where :math:`g(\cdot)` is :math:`g(x)=x\sigma(x)`, where :math:`\sigma(\cdot)` is the sigmoid function (:math:`\sigma(x)=(1+e^{-x})^{-1}`). :math:`\vec{e}_{ij}^{'}` is the updated edge features, :math:`\vec{e}_{ij}` is the original edge features.

The dimension of :math:`\tilde{e}_{ij}` depends on the choices of :math:`n` and :math:`l`, in total the vector length of :math:`\tilde{e}_{ij}` is :math:`n_{\text{max}}l_{\text{max}}`.

Regular edge features update
******************************

After considering the contribution of the many-body interactions, the edge features then can be updated conventionally:

.. math::

    \vec{e}_{ij}^{'} = \vec{e}_{ij} + \phi_e^k(\vec{v}_i\oplus\vec{v}_j\oplus\vec{e}_{ij}\oplus\vec{u})W_e^0\vec{e}_{ij}^0

where :math:`\oplus` represents the direct sum of the vectors, basically increase the dimension of the feature vectors, the meaning of the :math:`\oplus` can be seen in the :code:`learning_notes`. the expression for :math:`\vec{e}_{ij}^0` has been given above. In here :math:`\phi_e^k` is a gated function, which can be written as: 

.. math::

    \phi_e^k(\vec{x}) = (\mathcal{L}_g^k\circ\mathcal{L}_g^{k-1}\circ\cdots\circ\mathcal{L}_g^1)\odot(\mathcal{L}_{\sigma}^k\circ\mathcal{L}_g^{k-1}\circ\cdots\circ\mathcal{L}_g^1)(\vec{x})

where :math:`\mathcal{L}_g` is a linear layer with :math:`g(\cdot)` as the activation function, :math:`\mathcal{L}_{\sigma}` is a linear layer with :math:`\sigma(\cdot)` as the activation function. :math:`k` is the number of layers in the gated function.

Step 2: Node feature update
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The formula for node feature update is:

.. math::

    \vec{v}_i^{'} = \vec{v}_i + \sum_j\phi_v^{k}(\vec{v}_i\oplus\vec{v}_j\oplus\vec{e}_{ij}^{'}\oplus\vec{u})W_e^{0'}\vec{e}_{ij}^0

In here, in the paper they used :math:`\phi_e^{'}(\cdot)` as the updating function, but since it represents the update of the feature vector of the vertices, I used :math:`\phi_v^{k}(\cdot)`, where :math:`k` is the number of layers in the MLP.

Step 3: Global feature update
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

    \vec{u}^{'} = g(W_2^ug(W_1^u(\frac{1}{N_v}\sum_{i=1}^{N_v}\vec{v}_i\oplus\vec{u})+\vec{b}_1^u)+\vec{b}_2^u)

where :math:`N_v` is the number of vertices in the graph.

Step 4: Global pooling
^^^^^^^^^^^^^^^^^^^^^^^^

After message passing procedure, M3GNET uses feature vectors of nodes (vertices) to predict the properties of the material.

Extensive properties
******************************

For the extensive properties such as total energy, a three-layer gated MLP is used:

.. math::

    p_{ext} = \sum_i\phi_3(\{\vec{v}_i\})

where :math:`\phi_3(\cdot)` is a three-layer gated MLP, the number of hidden nodes for each layer is: :math:`[64, 64, 1]`, for the last layer, no activation function is used.

Intensive properties
******************************

For the intensive properties, the readout step was performed:

.. math::

    p_{int} = \xi_3(\sum_iw_i\xi_2(\vec{v}_i\oplus\vec{u}))

where the weights (:math:`w_i`) are calculated by:

.. math::

    w_i = \frac{\xi_3^{'}{(\vec{v}_i)}}{\sum_i\xi_3^{'}{(\vec{v}_i)}}

Loss function
--------------------

The loss function is:

.. math::

    L=l(e_{\text{pred}}, e_{\text{DFT}}) + w_fL_f(f_{\text{pred}}, f_{\text{DFT}}) + w_{\sigma}l(\sigma_{\text{pred}}, \sigma_{\text{DFT}})

where :math:`l(\cdot)` is the Huber loss function. 

Discussion
--------------------

- In M3GNET, during the message passing procedure, the length of edge and node features remain unchanged. Unlike in NequIP, where the length of the node features increases after each tensor product with the edge features.