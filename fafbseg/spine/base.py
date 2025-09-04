#    A collection of tools to interface with manually traced and autosegmented
#    data in FAFB.
#
#    Copyright (C) 2019 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
"""Collection of functions to query data from spine."""

import navis
import requests
import warnings

import numpy as np
import pandas as pd
import trimesh as tm

from abc import ABC
from io import BytesIO

from ..utils import make_iterable

use_pbars = True


class SpineService(ABC):
    """Abstract base class for spine services."""

    @property
    def info(self):
        """Return general info on given service."""
        if not hasattr(self, "_info"):
            url = self.makeurl("info")
            resp = self.session.get(url)
            resp.raise_for_status()
            self._info = resp.json()
        return self._info

    def urljoin(self, *args):
        """Join arguments into an url.

        Trailing but not leading slashes are stripped for each argument.

        """
        return "/".join(map(lambda x: str(x).rstrip("/"), args))

    def makeurl(self, *args):
        """Generate URL from base URL and args."""
        return self.urljoin(self.base_url, *args)


class FlyCacheService(SpineService):
    """Interface with cache services."""

    def __init__(self, base_url="https://services.itanna.io/app/flycache-dev"):
        """Init class."""
        self.base_url = base_url

    def get_L2_centroids(self, ids, token, as_array=False, chunksize=50, progress=True):
        """Fetch centroids of given l2 chunks.

        Coordinates are in nm.

        Parameter
        ---------
        ids :           iterable
                        Iterable with L2 IDs.
        token :         str
                        A ChunkedGraph/CAVE auth token.
        as_array :      bool
                        Determines output (see Returns).
        chunksize :     int
                        Query L2 IDs in chunks of this size.
        progress :      bool
                        Whether to show a progress bar or not.

        Returns
        -------
        array
                        If `as_array=True`: array in same order as queried `ids`.
                        Missing centroids are returns with coordinate (0, 0, 0).
        dict
                        If `as_array=False`: dictionary mapping ID to
                        centroid `{L2_ID: [x, y, z], ..}`.

        """
        url = self.makeurl("mesh/l2_centroid/flywire_fafb_production/")

        # Make sure we have an array of integers
        ids = make_iterable(ids, force_type=np.int64)

        with navis.config.tqdm(
            total=len(ids), desc="Fetching centroids", leave=False, disable=not progress
        ) as pbar:
            # First we will get everything that's cached in a single big query
            post = {"token": token, "query_ids": ids.tolist()}
            resp = self.session.post(url + "?cache_only=1", json=post)
            resp.raise_for_status()

            # Parse response
            centroids = {np.int64(k): v for k, v in resp.json().items()}

            # Filter to remaining IDs
            to_fetch = ids[~np.isin(ids, list(centroids))]

            # Update progress bar
            pbar.update(len(centroids))

            # Now go over the remaining indices in chunks
            for i in range(0, len(to_fetch), int(chunksize)):
                this_chunk = to_fetch[i : i + chunksize]
                post = {"token": token, "query_ids": this_chunk.tolist()}

                resp = self.session.post(url, json=post)
                resp.raise_for_status()

                # Parse response
                data = resp.json()

                # IDs will have been returned as strings
                centroids.update({np.int64(k): v for k, v in data.items()})

                pbar.update(len(this_chunk))

        if not as_array:
            return centroids

        centroids = np.array([centroids.get(i, [0, 0, 0]) for i in ids])

        return centroids


class SynapseService(SpineService):
    """Interface with synapse service on `services.itanna.io`.

    Kindly hosted by Eric Perlman and Davi Bock! Check out the online docs for
    available API endpoints:
    https://services.itanna.io/app/synapse-service/docs

    Parameters
    ----------
    base_url :      str
                    Base URL for the service on spine.

    """

    def __init__(self, base_url="https://services.itanna.io/app/synapse-service"):
        """Init class."""
        self.base_url = base_url
        self.session = requests.Session()

    @property
    def alignments(self):
        """Return available alignments of synapse data."""
        if not hasattr(self, "_alignments"):
            url = self.makeurl("alignments")
            resp = self.session.get(url)
            resp.raise_for_status()
            self._alignments = resp.json()
        return self._alignments

    @property
    def collections(self):
        """Return available collections of synapse data."""
        if not hasattr(self, "_collections"):
            url = self.makeurl("collections")
            resp = self.session.get(url)
            resp.raise_for_status()
            self._collections = resp.json()
        return self._collections

    @property
    def segmentations(self):
        """Return available segmentations the synapse data is mapped to."""
        if not hasattr(self, "_segmentations"):
            url = self.makeurl("segmentations")
            resp = self.session.get(url)
            resp.raise_for_status()
            self._segmentations = resp.json()
        return self._segmentations

    def validate_alignment(self, alignment):
        """Check if alignment exists."""
        available = [c.get("name", "NA") for c in self.alignments]
        if alignment not in available:
            raise ValueError(
                f"{alignment} not among available alignments: {','.join(available)}"
            )

    def validate_collection(self, collection):
        """Check if collection exists."""
        available = [c.get("name", "NA") for c in self.collections]
        if collection not in available:
            raise ValueError(
                f"{collection} not among available collections: {','.join(available)}"
            )

    def validate_segmentation(self, segmentation):
        """Check if alignment exists."""
        available = [c.get("name", "NA") for c in self.segmentations]
        if segmentation not in available:
            raise ValueError(
                f"{segmentation} not among available segmentations: "
                f"{','.join(available)}"
            )

    def get_synapse(self, synapse_id, collection):
        """Return all available info (pre/post IDs, location) for single synapse.

        Parameter
        ---------
        synapse_id :    int
                        ID of synapse to query.
        collection :    str
                        Collection to query for the synapse. See ``.collections``
                        for a list of available collections. Currently available:

                          - "buhmann2019": Buhmann synapses

        Returns
        -------
        dict

        """
        assert isinstance(synapse_id, (int, np.integer, str))

        self.validate_collection(collection)

        url = self.makeurl("collection", collection, "synapse", synapse_id, "info")
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json()

    def get_connectivity(
        self, segmentation_ids, segmentation, locations=False, nt_predictions=False
    ):
        """Fetch all connections from/to given segmentation ID(s).

        Parameters
        ----------
        segmentation_ids :  int | list thereof
                            Segmentation/root IDs to query.
        segmentation :      str
                            Which segmentation to search. Currently available:

                              - "fafb-ffn1-20190805" (segment IDs)
                              - "fafb-ffn1-20200412" (segment IDs)
                              - "flywire_supervoxels" (supervoxel IDs)
        locations :         bool
                            Whether to also fetch x/y/z locations for each
                            synapse.
        nt_predictions :    bool
                            Whether to also fetch neurotransmitter predictions
                            for each synapses.

        Returns
        -------
        pandas.DataFrame

        """
        # This always returns a numpy array
        segmentation_ids = navis.utils.make_iterable(segmentation_ids)

        # Check if segmentation actually exists
        self.validate_segmentation(segmentation)

        url = self.makeurl("segmentation", segmentation, "feather")

        param = []
        if locations:
            param.append("locations=true")
        if nt_predictions:
            param.append("nt=eckstein2020")

        if param:
            url += f"?{'&'.join(param)}"

        post = {"query_ids": segmentation_ids.tolist()}

        resp = self.session.post(url, json=post)
        resp.raise_for_status()

        # Read into DataFrame
        return pd.read_feather(BytesIO(resp.content))


class TransformService(SpineService):
    """Interface with transform service on `services.itanna.io`.

    Kindly hosted by Eric Perlman and Davi Bock! Check out the online docs for
    available API endpoints:
    https://services.itanna.io/app/transform-service/docs

    Parameters
    ----------
    base_url :      str
                    Base URL for the service on spine.

    """

    # Hard-coded data types for now. From:
    # https://github.com/bocklab/transform_service/blob/master/app/config.py
    # https://github.com/flyconnectome/transform_service/blob/master/app/config.py
    DTYPES = {
        "test": np.float32,
        "flywire_v1": np.float32,
        "flywire_v1_inverse": np.float32,
        "fanc_v4_to_v3": np.float32,
        "fafb-ffn1-20200412": np.uint64,
        "fafb-ffn1-20200412-gcs": np.uint64,
        "fafb-ffn1-20200412-zarr": np.uint64,
        "flywire_190410": np.uint64,
        "wclee_aedes_brain": np.uint64,
    }

    def __init__(self, base_url="https://services.itanna.io/app/transform-service"):
        """Init class."""
        self.base_url = base_url
        self.session = requests.Session()

    def validate_dataset(self, dataset):
        """Check if dataset exists for given service."""
        if dataset not in self.info:
            ds_str = ", ".join(self.info.keys())
            raise ValueError(f'"{dataset}" not among listed datasets: {ds_str}')

    def validate_mip(self, mip, dataset):
        """Validate mip for given dataset."""
        assert isinstance(mip, (int, np.integer))
        if mip < 0:
            available_scales = sorted(self.info[dataset]["scales"])
            mip = available_scales[-mip - 1]
        elif mip not in self.info[dataset]["scales"]:
            raise ValueError(
                f'mip {mip} not available for dataset "{dataset}". '
                f"Available scales: {self.info[dataset]['scales']}"
            )
        return mip

    def validate_output(self, x, on_fail="ignore"):
        """Validate output."""
        # If mapping failed will contain NaNs
        if on_fail != "ignore":
            is_nan = np.any(np.isnan(x), axis=1 if x.ndim == 2 else 0)
            if np.any(is_nan):
                msg = f"{is_nan.sum()} points failed to transform."
                if on_fail == "warn":
                    warnings.warn(msg)
                elif on_fail == "raise":
                    raise ValueError(msg)

    def to_voxels(self, x, dataset, coordinates="nm"):
        """Parse spatial data into voxels coordinates.

        Parameters
        ----------
        x :             np.ndarray (N, 3) | Neuron/List | mesh
                        Data to transform.
        coordinates :   "nm" | "voxel"
                        Units of the coordinates in ``x``.

        Returns
        -------
        vxl :           np.ndarray (N, 3)
                        x/y/z coordinates extracted from x and converted to
                        voxels.

        """
        assert coordinates in [
            "nm",
            "nanometer",
            "nanometers",
            "nanometre",
            "nanometres",
            "vxl",
            "voxel",
            "voxels",
        ]

        if isinstance(x, (navis.NeuronList, navis.TreeNeuron)):
            x = x.nodes[["x", "y", "z"]].values
        elif isinstance(x, (navis.Volume, navis.MeshNeuron, tm.Trimesh)):
            x = np.asarray(x.vertices)

        # At this point we are expecting a numpy array
        vxl = np.asarray(x)

        # Make sure data is now in the correct format
        if vxl.ndim != 2 or vxl.shape[1] != 3:
            raise TypeError(f"Expected (N, 3) array, got {x.shape}")

        # We need to convert to voxel coordinates
        # Note that we are rounding here to get to voxels.
        # This will have the most impact on the Z section.
        if coordinates not in ["vxl", "voxel", "voxels"]:
            # Make sure we are working with numbers
            # -> if dtype is "object" we will get errors from np.round
            if not np.issubdtype(vxl.dtype, np.number):
                vxl = vxl.astype(np.float64)

            # Convert to voxels
            vxl_size = self.info[dataset]["voxel_size"]
            vxl = vxl // vxl_size

        return vxl

    def get_offsets(
        self, x, transform, coordinates="nm", mip=-1, limit_request=10e9, on_fail="warn"
    ):
        """Transform coordinates.

        Parameters
        ----------
        x :             np.ndarray (N, 3) | Neuron/List | mesh
                        Data to transform.
        transform :     str
                        Dataset to use for transform. Currently available:

                         - 'flywire_v1': flywire to FAFB
                         - 'flywire_v1_inverse': FAFB to flywire
                         - 'fanc_v4_to_v3': FANC v4 to v3

                        See also ``.info`` for up-to-date info.
        mip :           int
                        Resolution of mapping. Lower = more precise but much slower.
                        Negative values start counting from the highest possible
                        resolution: -1 = highest, -2 = second highest, etc.
        coordinates :   "nm" | "voxel"
                        Units of the coordinates in ``x``.
        on_fail :       "warn" | "ignore" | "raise"
                        What to do if points failed to xform.

        Returns
        -------
        response :      np.ndarray
                        (N, 2) of dx and dy offsets.

        """
        assert on_fail in ["warn", "raise", "ignore"]

        # Check if dataset exists
        self.validate_dataset(transform)

        # Parse mip
        mip = self.validate_mip(mip, dataset=transform)

        # Extract voxels from x
        vxl = self.to_voxels(x, transform, coordinates=coordinates)

        # Generate URL
        url = self.makeurl(
            "transform/dataset",
            transform,
            "s",
            mip,
            "values_binary/format/array_float_Nx3",
        )

        # Make sure we don't exceed the maximum size for each request
        stack = []
        limit_request = int(limit_request)
        for ix in np.arange(0, vxl.shape[0], limit_request):
            this_vxl = vxl[ix : ix + limit_request]

            # Make request
            resp = self.session.post(
                url, data=this_vxl.astype(np.single).tobytes(order="C")
            )

            # Check for errors
            resp.raise_for_status()

            # Extract data
            data = np.frombuffer(resp.content, dtype=self.DTYPES[transform])

            # Reshape to [[dx1, dy1], [dx2, dy2], ...]
            data = data.reshape(this_vxl.shape[0], 2)
            stack.append(data)

        stack = np.concatenate(stack, axis=0)

        # See if any points failed to xform, and raise/warn if requested
        self.validate_output(stack, on_fail=on_fail)

        return stack

    def get_segids(
        self,
        x,
        segmentation,
        coordinates="nm",
        mip=-1,
        limit_request=10e9,
        on_fail="warn",
    ):
        """Fetch segmentation/supervoxel IDs.

        Parameters
        ----------
        x :             np.ndarray (N, 3) | Neuron/List | mesh
                        Coordinates to query segmentation IDs for.
        segmentation :  str
                        Segmentation to query. Currently available:

                         - 'fafb-ffn1-20200412'
                         - 'fafb-ffn1-20200412-gcs'
                         - 'fafb-ffn1-20200412-zarr'
                         - 'flywire_190410' (supervoxels)

        mip :           int
                        Resolution of mapping. Lower = more precise but much slower.
                        Negative values start counting from the highest possible
                        resolution: -1 = highest, -2 = second highest, etc.
        coordinates :   "nm" | "voxel"
                        Units of the coordinates in ``x``.
        limit_request : int
                        Max number of locations to query per request.
        on_fail :       "warn" | "ignore" | "raise"
                        What to do if points fail to return segmentation IDs.

        Returns
        -------
        response :      np.ndarray
                        (N, 2) of dx and dy offsets.

        """
        assert on_fail in ["warn", "raise", "ignore"]

        # Check if dataset exists
        self.validate_dataset(segmentation)

        # Parse mip
        mip = self.validate_mip(mip, dataset=segmentation)

        # Extract voxels from x
        vxl = self.to_voxels(x, segmentation, coordinates=coordinates)

        # Generate URL
        url = self.makeurl(
            "query/dataset",
            segmentation,
            "s",
            mip,
            "values_binary/format/array_float_Nx3",
        )

        # Make sure we don't exceed the maximum size for each request
        stack = []
        limit_request = int(limit_request)
        for ix in np.arange(0, vxl.shape[0], limit_request):
            this_vxl = vxl[ix : ix + limit_request]

            # Make request
            resp = self.session.post(
                url, data=this_vxl.astype(np.single).tobytes(order="C")
            )

            # Check for errors
            resp.raise_for_status()

            # Extract data
            data = np.frombuffer(resp.content, dtype=self.DTYPES[segmentation])

            stack.append(data)

        stack = np.concatenate(stack, axis=0)

        # See if any points failed to xform, and raise/warn if requested
        self.validate_output(stack, on_fail=on_fail)

        return stack


synapses = SynapseService()
transform = TransformService()
flycache = FlyCacheService()

lookup_services = [
    # Spine is also a look-up service
    transform,
    # We're hosting a look-up service on flyem too
    TransformService(base_url="https://flyem.mrc-lmb.cam.ac.uk/transform-service"),
]
