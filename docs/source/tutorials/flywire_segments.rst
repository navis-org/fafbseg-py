.. _flywire_segments:

Working with segmentation
=========================
For certain tasks it can be useful to query the flywire segmentation. Before we demonstrate this, a quick primer on terminology:

1. In flywire, the ID of a neuron (e.g. ``720575940618780781``) is called the "root ID".
2. Each root ID is a collection of "supervoxels". These supervoxels are the atomic, immutable units of the segmentation.
3. Everytime you edit a neuron (i.e. add or remove supervoxel by merging/splitting) it gets a new root ID.
    
Because of this dichotomy of IDs, you have to be explicit about whether you want root or supervoxel IDs.

Let's demonstrate this by running a realistic example: 

.. code:: ipython3

    import fafbseg
    import pymaid
    import navis
    
    # Connect to CATMAID
    rm = pymaid.connect_catmaid()
    
    # Load a neuron from CATMAID
    n = pymaid.get_neuron(16)
    
    # Transform neuron from FAFB14 to flywire voxels
    xf = navis.xform_brain(n, source='FAFB14', target='FLYWIREraw')


.. parsed-literal::

    INFO  : Global CATMAID instance set. Caching is ON. (pymaid)






.. parsed-literal::

    Transform path: FAFB14 -> FAFB14raw -> FLYWIREraw


Now we will fetch the root ID at each node of this CATMAID neuron

.. code:: ipython3

    xf.nodes['root_id'] = fafbseg.flywire.locs_to_segments(xf.nodes[['x','y','z']].values, root_ids=True)
    xf.nodes.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>node_id</th>
          <th>parent_id</th>
          <th>creator_id</th>
          <th>x</th>
          <th>y</th>
          <th>z</th>
          <th>radius</th>
          <th>confidence</th>
          <th>type</th>
          <th>root_id</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>72</td>
          <td>71</td>
          <td>2</td>
          <td>119526.679688</td>
          <td>37071.757812</td>
          <td>4258.0</td>
          <td>-0.1</td>
          <td>5</td>
          <td>slab</td>
          <td>720575940621835755</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1631</td>
          <td>1630</td>
          <td>2</td>
          <td>109709.054688</td>
          <td>33938.710938</td>
          <td>4923.0</td>
          <td>-0.1</td>
          <td>5</td>
          <td>slab</td>
          <td>720575940621835755</td>
        </tr>
        <tr>
          <th>2</th>
          <td>125055</td>
          <td>125054</td>
          <td>12</td>
          <td>110585.523438</td>
          <td>34993.304688</td>
          <td>4923.0</td>
          <td>-0.1</td>
          <td>5</td>
          <td>slab</td>
          <td>720575940621835755</td>
        </tr>
        <tr>
          <th>3</th>
          <td>217</td>
          <td>216</td>
          <td>2</td>
          <td>116653.437500</td>
          <td>35327.523438</td>
          <td>4404.0</td>
          <td>-0.1</td>
          <td>5</td>
          <td>slab</td>
          <td>720575940621835755</td>
        </tr>
        <tr>
          <th>4</th>
          <td>128575</td>
          <td>128576</td>
          <td>12</td>
          <td>122785.312500</td>
          <td>43073.925781</td>
          <td>3726.0</td>
          <td>-0.1</td>
          <td>5</td>
          <td>slab</td>
          <td>720575940621835755</td>
        </tr>
      </tbody>
    </table>
    </div>



Some of these root IDs are probably bycatch from imprecisely placed nodes. The following line of 
code counts how many times we have "hit" a given root ID:

.. code:: ipython3

    counts = xf.nodes.groupby('root_id').size().sort_values(ascending=False)
    counts.head(10)




.. parsed-literal::

    root_id
    720575940621835755    15550
    720575940608788840      562
    720575940628913983      103
    720575940623172447       83
    720575940616754529       56
    720575940521261247       53
    720575940595294780       32
    720575940627437275       31
    720575940619923352       18
    720575940638453494       15
    dtype: int64



Let's drop anything with less than 5 hits and load the segments into flywire:

.. code:: ipython3

    fafbseg.flywire.encode_url(segments=counts[counts >= 5].index.values, open_browser=True)




.. parsed-literal::

    'https://ngl.flywire.ai/?json_url=https://globalv1.flywire-daf.com/nglstate/5700142141276160'



If you open above URL, you will find that most of the collected flywire segments actually belong to this neuron and should be merged into a single neuron.
