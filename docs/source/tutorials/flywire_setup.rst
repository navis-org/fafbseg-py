.. _flywire_setup:

FlyWire setup
=============

In this tutorial you will learn all you need to know to get started
querying the FlyWire dataset.

Setting the FlyWire secret
--------------------------
Most data queries against FlyWire use ``cloudvolume`` or ``caveclient`` which
provide interfaces with FlyWire's chunkedgraph and annotation backends,
respectively. To be able to make those query you need to generate and store
an API token (or "secret"). This needs to be done only once.

Generate your secret
********************
Go to https://global.daf-apis.com/auth/api/v1/user/token and log in with
the account you use for FlyWire. You should then get bit of text with a token
that looks something like this:

.. code-block:: bat

  "ghd2nfk67kdndncf5kdmsqwo8kf23md6"

That's your token!

If (and only if) you get an empty ``[]`` you can visit
https://global.daf-apis.com/auth/api/v1/create_token instead which will
generate an entirely new token.

Saving the secret
*****************
Your token must then be saved to a file on your computer. ``fafbseg`` offers a
convenient function that does that for you (piggy-backing on ``caveclient``):

.. code-block:: python

  >>> from fafbseg import flywire
  >>> flywire.set_chunkedgraph_secret("ghd2nfk67kdndncf5kdmsqwo8kf23md6")


Alternatively, you can also manually create a file at
``~/.cloudvolume/secrets/global.daf-apis.com-cave-secret.json`` and add your
token like so:

.. code-block:: bat

  {
  "token: ""ghd2nfk67kdndncf5kdmsqwo8kf23md6"
  }

That's it, you're done! You should now be able to query the FlyWire dataset.
