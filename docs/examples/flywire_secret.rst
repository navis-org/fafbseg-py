.. _set_flywire_secret:

Setting the flywire secret
==========================
Many data queries against flywire use ``cloudvolume`` which provides an
interface with flywire's chunkedgraph data base. To be able to query that
backend you need to generate and save an API token (or "secret") similar to
CATMAID. This needs to be done only once.

Generate your secret
--------------------
Go to https://globalv1.flywire-daf.com/auth/api/v1/refresh_token and log in with
the account you use for flywire. You should then see a token that looks
something like this:

.. code-block:: bat

  "ghd2nfk67kdndncf5kdmsqwo8kf23md6"

That's your token!

Saving the secret
-----------------
Your token must then be must be saved in a file in
``~/.cloudvolume/secrets/chunkedgraph-secret.json``. That file needs to look
something like this:

.. code-block:: bat

  {
  "token: ""ghd2nfk67kdndncf5kdmsqwo8kf23md6"
  }

``fafbseg`` has a convenient function that does that for you:

.. code-block:: python

  >>> import fafbseg
  >>> fafbseg.flywire.set_chunkedgraph_secret("ghd2nfk67kdndncf5kdmsqwo8kf23md6")

That's it, your done! You should now be able to query the flywire dataset.
