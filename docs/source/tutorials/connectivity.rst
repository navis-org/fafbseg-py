.. _connectivity:

Fetching synaptic partners
==========================
In this notebook you will learn how to fetch synaptic partners of neurons of interest using the Buhmann *et al.* (2019) predictions. We will do this for flywire neurons as this is easier because we don't have to map a skeleton ID to Google segmentation ID(s) before querying synaptic partners. That being said, the connectivity functions in ``fafbseg`` usually come in pairs - one for ``flywire`` and one for ``google``.

Let's get started:

.. code:: ipython3

    import fafbseg
    
    # These are current root IDs of RHS DA1 uPNs
    da1_roots = [720575940638875637, 720575940615709888, 720575940606946929,
                720575940613822651, 720575940647428473, 720575940627629612,
                720575940617947045, 720575940633937527]



.. raw:: html

    <script type="text/javascript">
    window.PlotlyConfig = {MathJaxConfig: 'local'};
    if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
    if (typeof require !== 'undefined') {
    require.undef("plotly");
    requirejs.config({
        paths: {
            'plotly': ['https://cdn.plot.ly/plotly-latest.min']
        }
    });
    require(['plotly'], function(Plotly) {
        window._Plotly = Plotly;
    });
    }
    </script>



.. code:: ipython3

    # Get partners of these neurons
    cn_table = fafbseg.flywire.fetch_connectivity(da1_roots, style='catmaid')
    cn_table.head()








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
          <th>relation</th>
          <th>720575940638875637</th>
          <th>720575940615709888</th>
          <th>720575940606946929</th>
          <th>720575940613822651</th>
          <th>720575940647428473</th>
          <th>720575940627629612</th>
          <th>720575940617947045</th>
          <th>720575940633937527</th>
          <th>total</th>
        </tr>
        <tr>
          <th>id</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>720575940617169048</th>
          <td>upstream</td>
          <td>29.0</td>
          <td>40.0</td>
          <td>23.0</td>
          <td>6.0</td>
          <td>26.0</td>
          <td>14.0</td>
          <td>36.0</td>
          <td>29.0</td>
          <td>203.0</td>
        </tr>
        <tr>
          <th>720575940617947045</th>
          <td>upstream</td>
          <td>2.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>184.0</td>
          <td>0.0</td>
          <td>187.0</td>
        </tr>
        <tr>
          <th>720575940638875637</th>
          <td>upstream</td>
          <td>172.0</td>
          <td>3.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>180.0</td>
        </tr>
        <tr>
          <th>720575940615709888</th>
          <td>upstream</td>
          <td>1.0</td>
          <td>172.0</td>
          <td>0.0</td>
          <td>2.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3.0</td>
          <td>178.0</td>
        </tr>
        <tr>
          <th>720575940633937527</th>
          <td>upstream</td>
          <td>0.0</td>
          <td>3.0</td>
          <td>2.0</td>
          <td>7.0</td>
          <td>3.0</td>
          <td>0.0</td>
          <td>3.0</td>
          <td>155.0</td>
          <td>173.0</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    # Generate a neuroglancer link with the top 30 downstream partners
    top30ds = cn_table[cn_table.relation == 'downstream'].iloc[:30]
    fafbseg.flywire.encode_url(segments=top30ds.index.values, open_browser=True)




.. parsed-literal::

    'https://ngl.flywire.ai/?json_url=https://globalv1.flywire-daf.com/nglstate/5112574693605376'


