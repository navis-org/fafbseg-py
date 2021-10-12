.. _neuroglancer

Parsing and Generating FlyWire URLs
===================================
The modified neuroglancer used by flywire lets you share shortened URLs. 
We can both read these URLs to extract segment IDs and generate URLs 
to inject segment IDs or annotations.

First things first: if you haven't already, please generate and save your
:doc:`chunkedgraph secret<flywire_secret>` so that you can fetch flywire data.

Decoding URLs
*************

.. code:: ipython3

    import fafbseg
    
    # Paste shortened URL
    fafbseg.flywire.decode_url('https://ngl.flywire.ai/?json_url=https://globalv1.flywire-daf.com/nglstate/5658438042386432')




.. parsed-literal::

    {'position': [124266.1796875, 53184.04296875, 2593.12060546875],
     'annotations': [],
     'selected': ['720575940618780781']}



As you can see the exemplary URL only contains a single neuron ``720575940618780781`` and no annotations.
Note that by default :func:`fafbseg.flywire.decode_url` returns only parts of the neuroglancer "scene". 

.. code:: ipython3

    fafbseg.flywire.decode_url('https://ngl.flywire.ai/?json_url=https://globalv1.flywire-daf.com/nglstate/5658438042386432',
                               ret='full')




.. parsed-literal::

    {'layers': [{'source': 'precomputed://gs://microns-seunglab/drosophila_v0/alignment/image_rechunked',
       'type': 'image',
       'blend': 'default',
       'shaderControls': {},
       'name': 'Production-image'},
      {'source': 'graphene://https://prod.flywire-daf.com/segmentation/1.0/fly_v31',
       'type': 'segmentation_with_graph',
       'segments': ['720575940618780781'],
       'skeletonRendering': {'mode2d': 'lines_and_points', 'mode3d': 'lines'},
       'graphOperationMarker': [{'annotations': [], 'tags': []},
        {'annotations': [], 'tags': []}],
       'pathFinder': {'color': '#ffff00',
        'pathObject': {'annotationPath': {'annotations': [], 'tags': []},
         'hasPath': False}},
       'name': 'Production-segmentation_with_graph'}],
     'navigation': {'pose': {'position': {'voxelSize': [4, 4, 40],
        'voxelCoordinates': [124266.1796875, 53184.04296875, 2593.12060546875]}},
      'zoomFactor': 15.302007160698565},
     'perspectiveOrientation': [-0.0892081931233406,
      0.03333089128136635,
      -0.027877885848283768,
      0.9950647354125977],
     'perspectiveZoom': 4077.678405489736,
     'showSlices': False,
     'jsonStateServer': 'https://globalv1.flywire-daf.com/nglstate/post',
     'selectedLayer': {'layer': 'Production-segmentation_with_graph',
      'visible': True},
     'layout': 'xy-3d'}



You get much more info with ``ret='full'`` but probably not all that relevant to you.


Encoding URLs
*************
What about generating our own URLs? Easy! Lets start by recreating the same scene as we have above:

.. code:: ipython3

    url = fafbseg.flywire.encode_url(segments=[720575940618780781])
    url




.. parsed-literal::

    'https://ngl.flywire.ai/?json_url=https://globalv1.flywire-daf.com/nglstate/5753487916793856'



Opening that URL should load a scene containing only that one neuron. 

By-the-by: you can also open the URL straight away like so

.. code:: ipython3

    fafbseg.flywire.encode_url(segments=[720575940618780781], open_browser=True)

How about some colours?

.. code:: ipython3

    # Load neuron in red
    fafbseg.flywire.encode_url(segments=[720575940618780781], open_browser=True, seg_colors=['r'])




.. parsed-literal::

    'https://ngl.flywire.ai/?json_url=https://globalv1.flywire-daf.com/nglstate/5640077560512512'



We can also add x/y/z coordinates as annotations:

.. code:: ipython3

    fafbseg.flywire.encode_url(segments=[720575940618780781],
                               annotations=[[124266, 53184, 2500],
                                            [124266, 53184, 2600],
                                            [124266, 53184, 2700]],
                               open_browser=True, seg_colors=['r'])




.. parsed-literal::

    'https://ngl.flywire.ai/?json_url=https://globalv1.flywire-daf.com/nglstate/5679695580364800'



You can also add skeletons (e.g. loaded from CATMAID) as annotations but that unfortunately slows
down neuroglancer pretty quickly.

Check out :func:`fafbseg.flywire.encode_url` for full capabilities.
