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

"""Functions to fetch and set annotations in FlyWire using the annotation
framework and the materialization engine."""

import navis
import requests

import datetime as dt
import numpy as np
import pandas as pd

from requests_futures.sessions import FuturesSession
from typing import Optional
from functools import lru_cache
from pathlib import Path

from ..utils import make_iterable, download_cache_file, CACHE_DIR
from .utils import (get_cave_client, retry, get_chunkedgraph_secret, find_mat_version, inject_dataset, _is_valid_version)
from .segmentation import locs_to_segments, supervoxels_to_roots, is_latest_root

ANNOT_REPO_URL = "https://api.github.com/repos/flyconnectome/flywire_annotations"
FLYWIRE_ANNOT_URL = "https://github.com/flyconnectome/flywire_annotations/raw/main/supplemental_files/Supplemental_file1_annotations.tsv"

__all__ = ['get_somas',
           'get_materialization_versions',
           'get_annotations',
           'get_user_information',
           'list_annotation_tables',
           'create_annotation_table',
           'get_annotation_table_info',
           'delete_annotations', 'upload_annotations',
           'is_proofread',
           'search_community_annotations', 'search_annotations',
           'get_hierarchical_annotations']


PR_TABLE = {}
COMMUNITY_ANNOTATION_TABLE = "neuron_information_v2"
_annotation_tables = None
_user_information = {}


@inject_dataset(disallowed=['flat_630', 'flat_571'])
def is_proofread(x, materialization='auto', cache=True, validate=True, *,
                 dataset=None):
    """Test if neuron has been set to `proofread`.

    Parameters
    ----------
    x  :            int | list of int
                    Root IDs to check.
    materialization : "latest" | "live" | "auto" | int
                    Which materialization to check. If "latest" will use the
                    latest available one in the cave client.
    validate :      bool
                    Whether to validate IDs.
    cache :         bool
                    Use and update a locally cached version of the proofreading
                    table. Setting this to ``False`` will force fetching the
                    full table which is considerably slower. Does not apply
                    if `materialization='live'`.
    dataset :       "public" | "production" | "sandbox", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    proofread :     np.ndarray
                    Boolean array.

    """
    global PR_TABLE

    if not isinstance(x, (np.ndarray, set, list, pd.Series)):
        x = [x]

    # Force into array and convert to integer
    x = np.asarray(x, dtype=np.int64)

    # Check if any of the roots are outdated -> can't check those
    if validate and materialization == 'live':
        il = is_latest_root(x)
        if any(~il):
            print("At least some root ID(s) outdated and will therefore show up as "
                  f"not proofread: {x[~il]}")

    # Get available materialization versions
    client = get_cave_client(dataset=dataset)

    if materialization == 'latest':
        mat_versions = client.materialize.get_versions()
        materialization = max(mat_versions)
    elif materialization == 'auto':
        materialization = find_mat_version(x, dataset=dataset)

    if materialization == 'live':
        # For live materialization only do on-the-run queries
        table = client.materialize.live_query(table='proofreading_status_public_v1',
                                              timestamp=dt.datetime.utcnow(),
                                              filter_in_dict=dict(pt_root_id=x))
    elif isinstance(materialization, int):
        if cache:
            if materialization in PR_TABLE:
                table = PR_TABLE[materialization]
            else:
                table = client.materialize.query_table(table='proofreading_status_public_v1',
                                                       materialization_version=materialization)
                PR_TABLE[materialization] = table
        else:
            table = client.materialize.query_table(table='proofreading_status_public_v1',
                                                   filter_in_dict=dict(pt_root_id=x),
                                                   materialization_version=materialization)

    return np.isin(x, table.pt_root_id.values)


@inject_dataset(disallowed=['flat_630', 'flat_571'])
@retry
def is_materialized_root(id, materialization='latest', *, dataset=None):
    """Check if root existed at the time of materialization.

    Parameters
    ----------
    id :            int | list-like
                    Single ID or list of FlyWire (root) IDs.
    materialization : "latest" | int | "any"
                    Which materialization to check.
                     - "latest" will check against the latest materialization
                     - int will check against the given version
    dataset :       "public" | "production" | "sandbox", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    numpy array
                    Array of booleans

    See Also
    --------
    :func:`fafbseg.flywire.find_mat_version`
                    Use this to find a materialization version at which given
                    root IDs exist.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> flywire.is_materialized_root(720575940621039145)
    array([False])

    """
    id = make_iterable(id, force_type=np.int64)

    # Generaate array we can fill
    is_mat = np.zeros(len(id), dtype=bool)

    # Get root timestamps
    client = get_cave_client(dataset=dataset)
    ts_root_gen = client.chunkedgraph.get_root_timestamps(id)

    # Get timestamp at materalization
    ts_mat = client.materialize.get_timestamp(None if materialization == 'latest' else materialization)

    # Root IDs younger than the materialization can already be left at false
    older = ts_root_gen < ts_mat

    # Check which of the old-enough roots are still up-to-date
    if any(older):
        il = is_latest_root(id[older])

        if any(il):
            is_mat[np.where(older)[0][il]] = True

        if any(~il):
            # For those that aren't up-to-date anymore we have to make sure that
            # the were still "alive" at the materialization
            was_alive = []
            for i, ts in zip(id[older][~il], ts_root_gen[older][~il]):
                # Get the lineage graph from the root's creation right up to the
                # materialization
                G = client.chunkedgraph.get_lineage_graph(np.int64(i),
                                                          timestamp_past=ts,
                                                          timestamp_future=ts_mat, as_nx_graph=True)
                try:
                    # If there is a successor, this root was already dead
                    _ = next(G.successors(i))
                    was_alive.append(False)
                except StopIteration:
                    was_alive.append(True)
            was_alive = np.array(was_alive)
            if any(was_alive):
                is_mat[np.where(older)[0][was_alive]] = True

    return is_mat


@inject_dataset(disallowed=['flat_630', 'flat_571'])
def get_materialization_versions(*, dataset=None):
    """Fetch info on the available materializations.

    Parameters
    ----------
    dataset :   "public" | "production" | "sandbox" | "flat_630", optional
                Against which FlyWire dataset to query. If ``None`` will fall
                back to the default dataset (see
                :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    DataFrame

    """
    # Get/Initialize the CAVE client
    client = get_cave_client(dataset=dataset)

    # Get currently existing versions
    get_versions = retry(client.materialize.get_versions)
    versions = get_versions()

    # Fetch meta data
    meta = pd.DataFrame.from_records([client.materialize.get_version_metadata(v) for v in versions])

    # Reduce precision for timestamps to make more readable
    meta['expires_on'] = meta.expires_on.values.astype('datetime64[m]')
    meta['time_stamp'] = meta.time_stamp.values.astype('datetime64[m]')

    return meta.sort_values('version', ascending=False).reset_index(drop=True)


@inject_dataset(disallowed=['flat_630', 'flat_571'])
def create_annotation_table(name: str,
                            schema: str,
                            description: str,
                            voxel_resolution=[1, 1, 1],
                            *,
                            dataset=None):
    """Create annotation table.

    This is just a thin-wrapper around `CAVEclient.annotation.create_table`.

    Note that newly created tables will not show up as materialized until the
    next round of materialization. Unfortuntately, there is currently no way to
    query un-materialized tables.

    Existing tables can be browsed
    `here <https://prod.flywire-daf.com/annotation/views/aligned_volume/fafb_seung_alignment_v0>`_.

    Parameters
    ----------
    name :              str
                        Name of the table.
    schema :            str
                        Name of the schema. This determines what data we can put
                        in the table. Here are some useful ones:

                         - "bound_tag" contains a location ("pt") and a text
                           "tag"
                         - "contact" contains two locations ("sidea_pt" and
                           "sideb_pt"), a "size" (int) and a "ctr_pt" point
                         - "proofread_status" contains a location ("pt"), a
                           "valid_id" (int) and a "status" (str) field
                         - "cell_type_local" contains a location ("pt"), and
                           "cell_type" (str) and "classification_system" (str)
                           fields

                        See `here <https://globalv1.daf-apis.com/schema/views/>`
                        for a detailed list of all available schemas.
    description :       str
                        Human-readable description of what's in the table.
    voxel_resolution :  list of ints [x, y, z]
                        Voxel resolution points will be uploaded in. For example:
                         - [1,1,1] = coordinates are in nanometers
                         - [4,4,40] = coordinates are 4nm, 4nm, 40nm voxels
    dataset :           "public" | "production" | "sandbox", optional
                        Against which FlyWire dataset to query. If ``None`` will
                        fall back to the default dataset (see
                        :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    response
                        Server response if something went wrong.

    """
    # Get/Initialize the CAVE client
    client = get_cave_client(dataset=dataset)

    navis.utils.eval_param(name, name='name', allowed_types=(str, ))
    navis.utils.eval_param(schema, name='schema', allowed_types=(str, ))
    navis.utils.eval_param(description, name='description', allowed_types=(str, ))
    navis.utils.eval_param(voxel_resolution,
                           name='voxel_resolution',
                           allowed_types=(list, np.ndarray))

    if isinstance(voxel_resolution, np.ndarray):
        voxel_resolution = voxel_resolution.flatten().tolist()

    if len(voxel_resolution) != 3:
        raise ValueError('`voxel_resolution` must be list of [x, y, z], got '
                         f'{len(voxel_resolution)}')

    resp = client.annotation.create_table(table_name=name,
                                          schema_name=schema,
                                          description=description,
                                          voxel_resolution=voxel_resolution)

    if resp.content.decode() == name:
        print(f'Table "{resp.content.decode()}" successfully created.')
    else:
        print('Something went wrong, check response.')
        return resp


@inject_dataset(disallowed=['flat_630', 'flat_571'])
def list_annotation_tables(*, dataset=None):
    """Fetch available annotation tables.

    Parameters
    ----------
    dataset :       "public" | "production" | "sandbox", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    list

    """
    # Get/Initialize the CAVE client
    client = get_cave_client(dataset=dataset)

    an = client.annotation.get_tables()
    ma = client.materialize.get_tables()

    all_tables = list(set(an) | set(ma))

    df = pd.DataFrame(index=all_tables)
    df['annotation'] = df.index.isin(an)
    df['materialized'] = df.index.isin(ma)

    return df


@inject_dataset(disallowed=['flat_630', 'flat_571'])
def get_annotation_table_info(table_name: str,
                              *,
                              dataset=None):
    """Get info for given table.

    Parameters
    ----------
    table_name :    str
                    Name of the table.
    dataset :       "public" | "production" | "sandbox", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    info :              dict

    """
    # Get/Initialize the CAVE client
    client = get_cave_client(dataset=dataset)

    return client.annotation.get_table_metadata(table_name)


@inject_dataset(disallowed=['flat_630', 'flat_571'])
def get_annotations(table_name: str,
                    materialization='latest',
                    split_positions: bool = False,
                    drop_invalid: bool = True,
                    *,
                    dataset: Optional[str] = None,
                    **filters):
    """Get annotations from given table.

    Parameters
    ----------
    table_name :        str
                        Name of the table.
    materialization :   "live" | "latest" | int | bool | iterable
                        Which materialization version to fetch. You can also
                        provide an ID (int) for a specific materialization
                        version (see ``get_materialization_versions``).
                        If you provide a container of materialization versions
                        this function will search all of them and concatenate
                        the results (no deduplication).
                        Set to ``False`` to fetch the non-materialized version.
    split_positions :   bool
                        Whether to split x/y/z positions into separate columns.
    drop_invalid :      bool
                        Whether to drop invalidated (i.e. deleted or updated)
                        annotations.
    dataset :           "public" | "production" | "sandbox" | "flat_630", optional
                        Against which FlyWire dataset to query. If ``None`` will fall
                        back to the default dataset (see
                        :func:`~fafbseg.flywire.set_default_dataset`).
    **filters
                        Additional filter queries. See Examples. This works only
                        if ``materialization!=False``.

    Returns
    -------
    table :             pandas.DataFrame

    """
    if isinstance(materialization, (np.ndarray, tuple, list)):
        return pd.concat([get_annotations(table_name,
                                          materialization=v,
                                          split_positions=split_positions,
                                          drop_invalid=drop_invalid,
                                          dataset=dataset,
                                          **filters) for v in materialization],
                         axis=0)

    # Get/Initialize the CAVE client
    client = get_cave_client(dataset=dataset)

    navis.utils.eval_param(table_name, name='table_name', allowed_types=(str, ))

    if materialization in ('live', -1):  # internally we're treating -1 as live
        live_query = retry(client.materialize.live_query)
        data = live_query(table=table_name,
                          timestamp=dt.datetime.utcnow(),
                          split_positions=split_positions,
                          **filters)
    elif materialization:
        if materialization == 'latest':
            materialization = client.materialize.most_recent_version()

        query_table = retry(client.materialize.query_table)
        data = query_table(materialization_version=materialization,
                           table=table_name,
                           split_positions=split_positions,
                           **filters)
    else:
        raise ValueError('It is currently not possible to query the non-'
                         'materialized tables.')

    if drop_invalid and 'valid' in data.columns:
        data = data[data.valid == 't'].copy()
        data.drop('valid', axis=1, inplace=True)

    return data


@inject_dataset(disallowed=['flat_630', 'flat_571'])
def delete_annotations(table_name: str,
                       annotation_ids: list,
                       *,
                       dataset=None):
    """Delete annotations from table.

    Parameters
    ----------
    table_name :        str
                        Name of the table.
    annotation_ids :    int | list | np.ndarray | pandas.DataFrame
                        ID(s) of annotations to delete. If DataFrame must contain
                        an "id" column.
    dataset :           "public" | "production" | "sandbox", optional
                        Against which FlyWire dataset to query. If ``None`` will fall
                        back to the default dataset (see
                        :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    response :          str
                        Server response.

    """
    # Get/Initialize the CAVE client
    client = get_cave_client(dataset=dataset)

    navis.utils.eval_param(table_name, name='table_name', allowed_types=(str, ))
    navis.utils.eval_param(annotation_ids, name='annotation_ids',
                           allowed_types=(pd.DataFrame, list, np.ndarray, int))

    if isinstance(annotation_ids, pd.DataFrame):
        if 'id' not in annotation_ids.columns:
            raise ValueError('DataFrame must contain an "id" column')
        annotation_ids = annotation_ids['id'].values

    if isinstance(annotation_ids, np.ndarray):
        annotation_ids = annotation_ids.flatten().tolist()
    elif isinstance(annotation_ids, int):
        annotation_ids = [annotation_ids]

    resp = client.annotation.delete_annotation(table_name=table_name,
                                               annotation_ids=annotation_ids)

    print('Success! See response for details.')

    return resp


@inject_dataset(disallowed=['flat_630', 'flat_571'])
def upload_annotations(table_name: str,
                       data: pd.DataFrame,
                       *,
                       dataset=None):
    """Upload or update annotations to table.

    Parameters
    ----------
    table_name :    str
                    Name of the table.
    data :          pandas.DataFrame
                    Data to be uploaded. Must match the table's schema! If
                    'id' column exists, we assume that you want to update
                    existing annotations (i.e. rows in the table) with the
                    given IDs. See Examples for details.
    dataset :       "public" | "production" | "sandbox", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    response :          str
                        Server response.

    Examples
    --------
    Let's say we want to upload annotations to a table with the "bound_tag"
    schema. That schema requires a "pt" position and a "tag" (string) field.
    For all position fields, we need to provide them as a "{}_position" column
    (so "pt_position" in our example here) of x/y/z coordinates. Make sure
    they match the voxel resolution used for the table!

    >>> from fafbseg import flywire
    >>> import pandas as pd

    Generate the (mock) data we want to upload:

    >>> data = pd.DataFrame([])
    >>> data['pt_position'] = [[0,0,0], [100, 100, 100]]
    >>> data['tag'] = ['tag1', 'tag2']
    >>> data
           pt_position   tag
    0        [0, 0, 0]  tag1
    1  [100, 100, 100]  tag2

    Upload that data to a (fictional) table:

    >>> flywire.upload_annotations('my_table', data)

    To update annotations we can do the same thing but provide IDs:

    >>> # Look up IDs of annotations to update and add to DataFrame
    >>> data['id'] = [0, 1]
    >>> Make some changes to the data
    >>> data.loc[0, 'tag'] = 'new tag1'
    >>> flywire.upload_annotations('my_table', data)

    """
    # Get/Initialize the CAVE client
    client = get_cave_client(dataset=dataset)

    navis.utils.eval_param(table_name, name='table_name', allowed_types=(str, ))
    navis.utils.eval_param(data, name='data', allowed_types=(pd.DataFrame, ))

    if 'id' in data.columns:
        func = client.annotation.update_annotation_df
    else:
        func = client.annotation.post_annotation_df

    resp = func(table_name=table_name, df=data, position_columns=None)

    print('Success! See response for details.')

    return resp


@inject_dataset(disallowed=['flat_630', 'flat_571'])
def get_somas(x=None,
              materialization='auto',
              raise_missing=True,
              split_positions=False,
              *,
              dataset=None):
    """Fetch nuclei segmentation for given neuron(s).

    Parameters
    ----------
    x :                 int | list of ints | NeuronList, optional
                        FlyWire root ID(s) or neurons for which to fetch soma
                        infos. Use ``None`` to fetch complete list of annotated
                        nuclei. If neurons, will set their soma and soma radius
                        if one is found. Importantly, we assume that the neurons
                        are in nanometer space.
    materialization :   "auto" | "live" | "latest" | int | bool
                        Which materialization version to fetch. You can also
                        provide an ID (int) for a specific materialization
                        version (see ``get_materialization_versions``). Set to
                        False to fetch the non-materialized version.
    raise_missing :     bool
                        Only relevant if `materialization="auto"`: if True
                        (default) will complain if any of the query IDs can not
                        be found among the available materialization versions.
    split_positions :   bool
                        Whether to have separate columns for x/y/z position.
    dataset :           "public" | "production" | "sandbox", optional
                        Against which FlyWire dataset to query. If ``None`` will fall
                        back to the default dataset (see
                        :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    DataFrame
                        Pandas DataFrame with nucleu (see Examples). Root IDs
                        without a nucleus will be missing. The `rad_est` is
                        the estimated radius based on the widest point of the
                        nucleus. Take it with a grain of salt!

    Examples
    --------
    >>> from fafbseg import flywire
    >>> info = flywire.get_somas([720575940614131061])
    >>> info
            id valid   pt_supervoxel_id          pt_root_id     volume              pt_position  rad_est
    0  7393349     t  82827379285852979  720575940630194247  26.141245  [709888, 227744, 57160]   2260.0
    1  7415013     t  83038760463398837  720575940632921242  53.711176  [722912, 244032, 65200]   2640.0

    """
    if isinstance(x, navis.BaseNeuron):
        x = navis.NeuronList(x)

    filter_in_dict = None
    if not isinstance(x, type(None)):
        if isinstance(x, navis.NeuronList):
            root_ids = x.id.astype(np.int64)
        else:
            root_ids = make_iterable(x, force_type=np.int64)
        if materialization == 'auto':
            materialization = find_mat_version(root_ids,
                                               allow_multiple=True,
                                               raise_missing=raise_missing,
                                               dataset=dataset)
            if isinstance(materialization, np.ndarray):
                materialization = tuple(np.unique(materialization[materialization != 0]).tolist())
        filter_in_dict = {'pt_root_id': root_ids}

    nuc = get_annotations('nuclei_v1',
                        materialization=materialization,
                        split_positions=split_positions,
                        dataset=dataset,
                        filter_in_dict=filter_in_dict)
    nuc = nuc.drop_duplicates(['id', 'pt_root_id']).copy()

    # Add estimated radius based on nucleus
    if not nuc.empty:
        if not split_positions:
            start = np.vstack(nuc.bb_start_position)
            end = np.vstack(nuc.bb_end_position)
            nuc.drop(['bb_start_position', 'bb_end_position'],
                     inplace=True, axis=1)
        else:
            start_cols = [f'bb_start_position_{co}' for co in ['x', 'y', 'z']]
            end_cols = [f'bb_end_position_{co}' for co in ['x', 'y', 'z']]
            start = nuc[start_cols].values
            end = nuc[end_cols].values
            nuc.drop(start_cols + end_cols, inplace=True, axis=1)
        nuc['rad_est'] = np.abs(start - end).max(axis=1) / 2

        return nuc

        # If NeuronList, set their somas
        if isinstance(x, navis.NeuronList):
            if not split_positions:
                soma_pos = nuc.set_index('pt_root_id').pt_position.to_dict()
            else:
                soma_pos = dict(zip(nuc.pt_root_id, nuc[['pt_position_x', 'pt_position_y', 'pt_position_z']].values))
            soma_rad = nuc.set_index('pt_root_id').rad_est.to_dict()
            for n in x:
                # Skip if no soma found
                if np.int64(n.id) not in soma_pos:
                    continue

                if isinstance(n, navis.TreeNeuron):
                    n.soma = n.snap(soma_pos[np.int64(n.id)])[0]
                    n.nodes.loc[n.nodes.node_id == n.soma, 'radius'] = soma_rad[np.int64(n.id)]
                    n._clear_temp_attr()  # not sure why but this is necessary for some reason
                    n.reroot(n.soma, inplace=True)
                elif isinstance(n, navis.MeshNeuron):
                    n.soma_pos = soma_pos[np.int64(n.id)]
    else:
        # Make sure the column exist even in an empty table
        nuc['rad_est'] = []

    # Sorting by radius makes sure that the small false-positives end up at the
    # bottom of the list
    return nuc.sort_values('rad_est', ascending=False)


def submit_cell_identification(x, split_tags=False, validate=True,
                               skip_existing=False, max_threads=4,
                               progress=True):
    """Submit a identification for given cells.

    Requires access to production dataset. Use this bulk submission of cell
    identification with great care!

    Parameters
    ----------
    x :             pandas.DataFrame
                    Must have the following columns:
                      - `valid_id` (or `root_id`,  `root` or `id`)
                        contains the current root ID
                      - `x`, `y`, `z` (or `pos_x`, `pos_y`, `pos_z`) contain
                        coordinates mapping to that root (must be in voxel space)
                      - `tag` (or `tags`) must be a comma-separated string of
                        tags
                      - `user_id` (optional) if you want to submit for someone
                        else
    split_tags :    bool
                    If True, will split the comma-separated tags into
                    individual tags.
    validate :      bool
                    Whether to run some sanity checks on annotations
                    (recommended)!
    skip_existing : bool
                    Skip annotation if it already exist on given neuron. Note
                    that this is limited by the 24h delay until annotations
                    materialize.
    max_threads :   int
                    Number of parallel submissions.
    progress :      bool
                    If True, show progress bar.

    Returns
    -------
    submitted :     pandas.DataFrame
                    A list of tags the were supposed to be submitted. Includes a
                    `success` and an `error` column. You can use this to
                    retry submission in case of errors.

    """
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f'Expected DataFrame, got {type(x)}')

    REQ_COLS = (('valid_id', 'root_id', 'root', 'id'),
                ('x', 'pos_x'),
                ('y', 'pos_y'),
                ('z', 'pos_z'),
                ('tag', 'tags'))
    for c in REQ_COLS:
        if isinstance(c, tuple):
            # Check that at least one option exits
            if not any(np.isin(c, x.columns)):
                raise ValueError(f'`x` must contain one of these column: {c}')
            # Rename so we always find the first possible option
            if c[0] not in x.columns:
                for v in c[1:]:
                    if v in x.columns:
                        x = x.rename({v: c[0]}, axis=1)
        else:
            if c not in x:
                raise ValueError(f'Missing required column: {c}')

    if validate:
        roots = locs_to_segments(x[['x', 'y', 'z']].values)

        is_zero = roots == 0
        if any(is_zero):
            raise ValueError(f'{is_zero.sum()} xyz coordinates map to root ID 0')

        mm = roots != x.valid_id.astype(np.int64)
        if any(mm):
            raise ValueError(f'{mm.sum()} xyz coordinates do not map to the '
                             'provided `valid_id`')

    session = requests.Session()
    future_session = FuturesSession(session=session, max_workers=max_threads)

    token = get_chunkedgraph_secret()
    session.headers['Authorization'] = f"Bearer {token}"

    if skip_existing:
        existing = _get_cell_type_table(update_ids=True, dataset='production')

    futures = {}
    skipped = []
    url = f'https://prod.flywire-daf.com/neurons/api/v1/submit_cell_identification'

    for i, (_, row) in enumerate(x.iterrows()):
        if split_tags:
            tags = row.tag.split(',')
        else:
            tags = [row.tag]

        if skip_existing:
            existing_tags = existing.loc[existing.pt_root_id == np.int64(row.valid_id),
                                         'tag'].unique()

        for tag in tags:
            if skip_existing and tag in existing_tags:
                skipped.append([(i, tag)])
                continue

            post = dict(valid_id=str(row.valid_id),
                        location=f'{row.x}, {row.y}, {row.z}',
                        tag=tag,
                        action='single',
                        user_id=row.get('user_id', ''))

            f = future_session.post(url, data=post)
            futures[f] = [i, row.valid_id, row.x, row.y, row.z, tag]

    if len(skipped):
        navis.config.logger.info(f'Skipped {len(skipped)} existing annotations.')

    # Get the responses
    resp = [f.result() for f in navis.config.tqdm(futures,
                                                  desc='Submitting',
                                                  disable=not progress or len(futures) == 1,
                                                  leave=False)]

    submitted = []
    for r, f in zip(resp, futures):
        submitted.append(futures[f])
        try:
            r.raise_for_status()
        except BaseException as e:
            submitted[-1] += [False, str(e)]
            continue

        if 'Success' not in r.text:
            submitted[-1] += [False, r.text]
            continue

        submitted[-1] += [True, None]

    submitted = pd.DataFrame(submitted,
                             columns=['row_ix', 'valid_id', 'x', 'y', 'z', 'tag',
                                      'success', 'errors'])

    if not all(submitted.success):
        failed_ix = submitted[~submitted.success].row_ix.unique().astype(str)
        navis.config.logger.error(f'Encountered {(~x.success).sum()} errors '
                                  'while marking cells as proofread. Please '
                                  'see the `errors` columns in the returned '
                                  'dataframe for details. Affected rows in '
                                  f'original dataframe: {", ".join(failed_ix)}')
    else:
        navis.config.logger.info('SUCCESS!')

    return submitted


@inject_dataset(disallowed=['sandbox'])
def search_annotations(x,
                       exact=False,
                       case=False,
                       regex=True,
                       clear_cache=False,
                       materialization='auto',
                       *,
                       dataset=None):
    """Search hierarchical annotations (super class, cell class, cell type, etc).

    Annotations stem from Schlegel et al 2023 (bioRxiv); corresponds to entries
    in the "Classification" column in Codex.

    This function downloads and caches the supplemental annotation table hosted
    on Github at https://github.com/flyconnectome/flywire_annotations/. Updates
    to the Github repository will trigger an update of the cached file. If you
    find any issues with the annotations, please open an issue on Github.

    Parameters
    ----------
    x :         str | int | Neuron/List | list of ints | None
                Term (str) or root ID(s) to search for. See examples for details.
                Use `None` to return all annotations.
    exact :     bool
                Whether term must be an exact match. For example, if
                ``exact=False`` (default), 'sensory' will match to e.g.
                'sensory,olfactory' or 'AN sensory'.
    case :      bool
                If True (default = False), search for term will be case
                sensitive.
    regex :     bool
                Whether to interpret term as regex.
    clear_cache : bool
                If True, will clear the cached annotation table(s).
    materialization :   "auto" | "live" | "latest" | int | bool
                Which materialization version to search:
                 - "auto": if `x` is root ID(s), will try to find a version
                   at which all root IDs co-existed; if `x` is string will use
                   "latest" (see below)
                 - "latest": uses the latest _already cached_ materialization
                 - integer: specifies a materialization version
                 - "live": looks up the most recent root IDs from the supervoxels
    dataset :   "public" | "production", optional
                Against which FlyWire dataset to query. If ``None`` will fall
                back to the default dataset (see also
                :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    pandas.DataFrame
                If `x` is root IDs
                Pandas DataFrame with annotations matching ``x``. Coordinates
                are in 4x4x40nm voxel space.

    See Also
    --------
    :func:`~fafbseg.flywire.get_hierarchical_annotations`
                Use this to load get a table with all hierarchical annotations.
    :func:`~fafbseg.flywire.search_community_annotations`
                Use this to search the community annotations.

    Examples
    --------

    Find info for given root ID(s)

    >>> an = flywire.search_annotations(720575940628857210)
    >>> an.iloc[0]
    supervoxel_id               78112261444987077
    root_id                    720575940628857210
    pos_x                                  109306
    pos_y                                   50491
    pos_z                                    3960
    soma_x                               104904.0
    soma_y                                47464.0
    soma_z                                 5461.0
    nucleus_id                          2453924.0
    flow                                intrinsic
    super_class                           central
    cell_class                                NaN
    cell_sub_class                            NaN
    cell_type                                 NaN
    hemibrain_type                          PS180
    ito_lee_hemilineage            SMPpv2_ventral
    hartenstein_hemilineage           CP1_ventral
    morphology_group             SMPpv2_ventral_3
    top_nt                          acetylcholine
    top_nt_conf                          0.914499
    side                                     left
    nerve                                     NaN
    fbbt_id                         FBbt_20001935
    status                                    NaN

    Search for a cell type

    >>> ps009 = flywire.search_annotations('PS009', exact=True)

    Use "colum:value" to search for a specific field

    >>> phn = flywire.search_annotations('nerve:PhN')

    Use regex to refine search (here we try finding all "PSXXX" hemibrain types)

    >>> all_ps = flywire.search_annotations('hemibrain_type:PS[0-9]{3}', regex=True)

    """
    # See if ``x`` is a root ID as string
    if isinstance(x, str):
        try:
            x = np.int64(x)
        except ValueError:
            pass

    if isinstance(x, (int, np.int64, np.int32)):
        x = np.array([x])
    elif isinstance(x, (list, tuple)):
        x = np.array(x)
    elif isinstance(x, navis.NeuronList):
        x = x.id
    elif isinstance(x, navis.BaseNeuron):
        x = np.array([x.id])

    # Make sure root IDs are integers
    if isinstance(x, np.ndarray):
        try:
            x = x.astype(np.int64)
        except ValueError:
            pass

    if materialization == 'auto':
        # If query is not a bunch of root IDs, just use the latest version
        if not isinstance(x, np.ndarray) or x.dtype not in (int, np.int64):
            materialization = 'latest'
        else:
            # First check among the available versions
            cached_versions = _get_cached_annotation_materializations()
            if len(cached_versions):
                for version in sorted(cached_versions)[::-1]:
                    if _is_valid_version(ids=x, version=version, dataset=dataset):
                        materialization = version
                        print(f'Using cached materialization version {version}')
                        break
            else:
                materialization = find_mat_version(x, raise_missing=False, dataset=dataset)

    if materialization == 'latest':
        # Map to the latest cached version
        cached_versions = _get_cached_annotation_materializations()
        available_version = get_cave_client(dataset=dataset).materialize.get_versions()
        available_and_cached = cached_versions[np.isin(cached_versions, available_version)]
        if len(available_and_cached):
            materialization = sorted(available_and_cached)[-1]
        else:
            materialization = sorted(available_version)[-1]
        print(f'Using materialization version {materialization}')

    # Grab the table at the requested materialization
    ann = get_hierarchical_annotations(mat=materialization,
                                       dataset=dataset,
                                       force_reload=clear_cache)

    # If no query term, we'll just return the whole table
    if x is None:
        return ann

    # Search for given tag if `x` is string
    if isinstance(x, str):
        if ':' in x:
            col, x = x.split(':')
            if col not in ann.columns:
                raise ValueError(f'Annotation table has no column called "{col}"')
            cols = [col]
        else:
            # Get all string columns
            dtypes = ann.dtypes
            cols = dtypes[(dtypes == object) & ~dtypes.index.str.contains('root')].index

        filter = np.zeros(len(ann), dtype=bool)
        for col in cols:
            if not exact:
                filter[ann[col].str.contains(x, case=case, regex=regex, na=False)] = True
            elif not regex:
                filter[ann[col].str == x] = True
            else:
                filter[ann[col].str.match(x, case=case, na=False)] = True

        # Filter
        ann = ann.loc[filter]
    else:
        ann = ann[ann['root_id'].isin(x)]

    # Return copy to avoid setting-on-copy warning
    return ann.copy()


@inject_dataset()
def get_hierarchical_annotations(mat=None,
                                 check_updates=True,
                                 force_reload=False,
                                 verbose=True,
                                 *,
                                 dataset=None):
    """Download (and cache) hierarchical annotations.

    Annotations stem from Schlegel et al 2023 (bioRxiv); corresponds to entries
    in the "Classification" column in Codex.

    This function downloads and caches the supplemental annotation table hosted
    on Github at https://github.com/flyconnectome/flywire_annotations/. Updates
    to the Github repository will trigger an update of the cached file. If you
    find any issues with the annotations, please open an issue on Github.

    Parameters
    ----------
    mat :       "live" | "latest" | int, optional
                Which materialization to fetch:
                 - if int, will add a "root_{mat}" column with the respective IDs
                 - if "live", will update the "root_id" column to be current
                 - if `None` will return cached table as is
    check_updates : bool
                If True, will check the Github repository for updates of the
                hierarchical annotations and download again if necessary.
    force_reload :  bool
                If True, will force fresh download of file even if already cached
                locally.

    Returns
    -------
    DataFrame

    See Also
    --------
    :func:`~fafbseg.flywire.search_annotations`
                Use this to search for annotations for given neurons.
    :func:`~fafbseg.flywire.search_community_annotations`
                Use this to search the community annotations.

    """
    # To-Do:
    # - add option to check out specific release from Github repo

    fp = Path(CACHE_DIR).expanduser().absolute() / Path(FLYWIRE_ANNOT_URL).name

    # If file already exists, check if we need to refresh the cache
    if fp.exists() and check_updates and not force_reload:
        r = requests.get(ANNOT_REPO_URL)
        try:
            r.raise_for_status()
        except BaseException:
            print("Failed to check annotation repo for updates")
        # Last time anything was committed to the repo
        last_upd = dt.datetime.fromisoformat(r.json()["pushed_at"][:-1])
        # Last time the local file was modified
        last_mod = dt.datetime.fromtimestamp(fp.stat().st_mtime)
        if last_mod < last_upd:
            force_reload = True
            if verbose:
                print(f"Updating annotation table from {ANNOT_REPO_URL}")

    # This will only download file if it either doesn't exist or needs updating
    fp = download_cache_file(
        FLYWIRE_ANNOT_URL, force_reload=force_reload, verbose=verbose
    )

    # Read the actual table
    table = pd.read_csv(fp, sep="\t", low_memory=False)

    # Turn supervoxel and all root ID columns into integers
    dtypes = {'supervoxel_id': np.int64}
    dtypes.update({c: np.int64 for c in table.columns if str(c).startswith('root_')})
    table = table.astype(dtypes)

    # Map to the latest version
    if mat == 'latest':
        client = get_cave_client()
        mat = sorted(client.materialize.get_versions())[-1]

    # If mat is live we need to check for outdated IDs
    save = False
    if mat in ("live", "current") and (dataset == 'production'):
        to_update = ~is_latest_root(table.root_id, progress=False)
        if any(to_update):
            if verbose:
                print(
                    "Updating root IDs for hierarchical annotations... ",
                    end="",
                    flush=True,
                )
            table.loc[to_update, "root_id"] = supervoxels_to_roots(
                table.supervoxel_id.values[to_update], progress=False
            )
            save = True
            root_col = 'root_id'
    # If `mat` is not None
    elif mat:
        root_col = f"root_{mat}"
        if root_col not in table.columns:
            if verbose:
                print(
                    f"Updating root IDs for hierarchical annotations at mat '{mat}'... ",
                    end="",
                    flush=True,
                )
            save = True
            table[root_col] = supervoxels_to_roots(table.supervoxel_id,
                                                   timestamp=f"mat_{mat}",
                                                   progress=False)

    # If me made changes (i.e. updated the root ID column) save this back to disk
    # so we don't have to do it again
    if save:
        table.to_csv(fp, index=False, sep='\t')
        if verbose:
            print('Done.', flush=True)

    # Make sure "root_id" corresponds to the correct materialization and drop
    # all others to avoid confusion
    table['root_id'] = table[root_col]
    table = table.drop([c for c in table.columns if ('root_' in c) and (c != 'root_id')],
                       axis=1)

    return table


def _get_cached_annotation_materializations():
    """Which materialization versions have been cached for the annotation table."""
    fp = Path(CACHE_DIR).expanduser().absolute() / Path(FLYWIRE_ANNOT_URL).name

    # If file already exists, check if we need to refresh the cache
    if not fp.exists():
        return []

    # Read the actual table
    table = pd.read_csv(fp, sep="\t", low_memory=False, nrows=1)

    # Parse root ID columns
    mats = []
    for col in table.columns:
        if not col.startswith('root_'):
            continue
        try:
            this_mat = int(col.replace('root_', ''))
            mats.append(this_mat)
        except ValueError:
            pass

    return np.array(mats)


def get_user_information(user_ids, field=None):
    """Fetch (and cache) user information (name, affiliation, etc.) from their IDs.

    Parameters
    ----------
    user_ids :  list of integers
                List of IDs for which to find user information.
    field :     str, optional
                If provided will only return given field (e.g "name").

    Returns
    -------
    list

    """
    # Get IDs missing from cache
    missing = [i for i in user_ids if i not in _user_information]

    # Fetch info for missing IDs and update cache
    if len(missing):
        client = get_cave_client()
        _user_information.update({r['id']: r for r in client.auth.get_user_information(missing)})

    if field is None:
        return [_user_information[i] for i in user_ids]
    else:
        return [_user_information[i][field] for i in user_ids]


@inject_dataset(disallowed=['sandbox'])
def search_community_annotations(x,
                   exact=False,
                   case=False,
                   regex=True,
                   clear_cache=False,
                   materialization='auto',
                   *,
                   dataset=None):
    """Search community cell identification annotations for given term/root IDs.

    This function loads and caches the cell type information table, i.e. the
    first call might take a while but subsequent calls should be very fast.

    Parameter
    ---------
    x :         str | int | Neuron/List | list of ints | None
                Term (str) or root ID(s) to search for. Set to `None` to fetch
                all annotations.
    exact :     bool
                Whether term must be an exact match. For example, if
                ``exact=False`` (default), 'sensory' will match to e.g.
                'sensory,olfactory' or 'AN sensory'.
    case :      bool
                If True (default = False), search for term will be case
                sensitive.
    regex :     bool
                Whether to interpret term as regex.
    clear_cache : bool
                If True, will clear the cached annotation table(s).
    materialization :   "auto" | "live" | "latest" | int | bool
                Which materialization version to search. You can also
                provide an ID (int) for a specific materialization
                version (see ``get_materialization_versions``). "auto" is only
                relevant if `x` is a list of root IDs.
    dataset :   "public" | "production", optional
                Against which FlyWire dataset to query. If ``None`` will fall
                back to the default dataset (see also
                :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    pandas.DataFrame
                If `x` is root IDs
                Pandas DataFrame with annotations matching ``x``. Coordinates
                are in 4x4x40nm voxel space.

    See Also
    --------
    :func:`~fafbseg.flywire.search_annotations`
                Use this to search through the hiearchical annotations.
    :func:`~fafbseg.flywire.get_hierarchical_annotations`
                Use this to load get a table with all hierarchical annotations.

    Examples
    --------

    Search for annotations for given root ID(s)

    >>> an = flywire.search_community_annotations(720575940628857210)
    >>> an.iloc[0]
    id                                             46699
    created             2022-04-20 17:26:55.132886+00:00
    superceded_id                                    NaN
    pt_position_x                                 419980
    pt_position_y                                 189644
    pt_position_z                                 217360
    tag                           unclassified_IN_FW_112
    user                                 Stefanie Hampel
    user_id                                          125
    pt_supervoxel_id                   77830580511126708
    pt_root_id                        720575940628857210

    Search for all tags matching a given pattern

    >>> ps009 = flywire.search_community_annotations('PS009')
    >>> ps009
                id                          created  ...      supervoxel_id             root_id
    132997  150510 2023-09-11 18:15:40.698227+00:00  ...  82194060924379865  720575940624142995
    134029  150511 2023-09-11 18:15:51.786862+00:00  ...  82194060924285875  720575940640133941
    134456  150514 2023-09-11 18:27:45.745936+00:00  ...  82123692180213748  720575940624142995
    134502  150515 2023-09-11 18:27:55.483635+00:00  ...  82053392222021289  720575940640133941
    154788  156322 2023-09-13 20:18:26.050154+00:00  ...  82123829686174290  720575940615799697

    """
    # See if ``x`` is a root ID as string
    if isinstance(x, str):
        try:
            x = np.int64(x)
        except ValueError:
            pass

    if isinstance(x, (int, np.int64, np.int32)):
        x = np.array([x])
    elif isinstance(x, (list, tuple)):
        x = np.array(x)
    elif isinstance(x, navis.NeuronList):
        x = x.id
    elif isinstance(x, navis.BaseNeuron):
        x = np.array([x.id])

    # Make sure root IDs are integers
    if isinstance(x, np.ndarray):
        try:
            x = x.astype(np.int64)
        except ValueError:
            pass

    if materialization == 'auto':
        if isinstance(x, np.ndarray):
            materialization = find_mat_version(x, raise_missing=False, dataset=dataset)
        else:
            materialization = 'latest'

    if clear_cache:
        _get_cell_type_table.cache_clear()

    # Grab the table at the requested materialization
    ct = _get_cell_type_table(dataset=dataset,
                                 split_positions=True,
                                 materialization=materialization)

    # If no query term, we'll just return the whole
    if x is None:
        return ct

    # Search for given tag if `x` is string
    if isinstance(x, str):
        if not exact:
            ct = ct[ct.tag.str.contains(x, case=case, regex=regex)]
        elif not regex:
            ct = ct[ct.tag == x]
        else:
            ct = ct[ct.tag.str.match(x, case=case)]

        # Avoid setting-on-copy warning
        ct = ct.copy()

        # Rename columns to make it less clunky to work with
        ct = ct.rename({'pt_position_x': 'pos_x',
                        'pt_position_y': 'pos_y',
                        'pt_position_z': 'pos_z',
                        'pt_root_id': 'root_id',
                        'pt_supervoxel_id': 'supervoxel_id'},
                        axis=1)

        # Convert from nm to voxel space
        ct[['pos_x', 'pos_y', 'pos_z']] //= [4, 4, 40]
    else:
        ct = ct[ct.pt_root_id.isin(x)]

    if not ct.empty:
        name_map = dict(zip(ct.user_id.unique(),
                            get_user_information(ct.user_id.unique(), field='name')))
        ct.insert(ct.columns.tolist().index('user_id'),
                  'user',
                  ct.user_id.map(name_map))

    return ct


@lru_cache
def _get_cell_type_table(dataset, materialization, split_positions=False, verbose=True):
    """Fetch (and cache) annotation tables."""
    if materialization == 'latest':
        versions = get_cave_client(dataset=dataset).materialize.get_versions()
        materialization = sorted(versions)[-1]

    if verbose:
        print(f'Caching community annotations for materialization version "{materialization}"...',
              end='', flush=True)
    table = get_annotations(table_name=COMMUNITY_ANNOTATION_TABLE,
                            dataset=dataset,
                            split_positions=split_positions,
                            materialization=materialization)
    if verbose:
        print(' Done.')
    return table


def mark_cell_completion(x, validate=True, skip_existing=True,
                         max_threads=4, progress=True):
    """Submit proofread status for given cell.

    Use this bulk submission of proofreading status with great care! Requires
    access to the production dataset.

    Parameters
    ----------
    x :             pandas.DataFrame
                    Must have the following columns:
                      - `valid_id` contains the current root ID
                      - `x`, `y`, `z` contain coordinates mapping to that root
                        (must be in voxel space)
                      - `user_id` (optional) if you want to submit for someone
                        else
    validate :      bool
                    Whether to run some sanity checks on annotations
                    (recommended)!
    skip_existing : bool
                    If True, will skip neurons that have already been marked
                    proofread.
    max_threads :   int
                    Number of parallel submissions.
    progress :      bool
                    If True, show progress bar.

    Returns
    -------
    response :      pandas.DataFrame
                    A copy of ``x`` with a `success` and an `error` column.

    """
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f'Expected DataFrame, got {type(x)}')

    REQ_COLS = (('valid_id', 'root_id', 'root', 'id'),
                ('x', 'pos_x'),
                ('y', 'pos_y'),
                ('z', 'pos_z'))
    for c in REQ_COLS:
        if isinstance(c, tuple):
            # Check that at least one option exits
            if not any(np.isin(c, x.columns)):
                raise ValueError(f'`x` must contain one of these column: {c}')
            # Rename so we always find the first possible option
            if c[0] not in x.columns:
                for v in c[1:]:
                    if v in x.columns:
                        x = x.rename({v: c[0]}, axis=1)
        else:
            if c not in x:
                raise ValueError(f'Missing required column: {c}')

    if validate:
        roots = locs_to_segments(x[['x', 'y', 'z']].values)

        is_zero = roots == 0
        if any(is_zero):
            raise ValueError(f'{is_zero.sum()} xyz coordinates map to root ID 0')

        mm = roots != x.valid_id.astype(np.int64)
        if any(mm):
            raise ValueError(f'{mm.sum()} xyz coordinates do not map to the '
                             'provided `valid_id`')

        u, cnt = np.unique(roots, return_counts=True)
        if any(cnt > 1):
            raise ValueError(f'{(cnt > 1).sum()} root IDs are duplicated: '
                             f'{u[cnt > 1]}')

    if skip_existing:
        pr = is_proofread(x.valid_id.values)
        if any(pr):
            navis.config.logger.info(f'Dropping {pr.sum()} neurons that have '
                                     'already been proofread')
            x = x[~pr]
        if x.empty:
            navis.config.logger.info('Looks like all neurons have already '
                                     'been set to proofread')
            return pd.DataFrame(columns=['valid_id', 'x', 'y', 'z', 'success', 'errors'])

    if 'user_id' not in x.columns:
        x['user_id'] = ''

    session = requests.Session()
    future_session = FuturesSession(session=session, max_workers=max_threads)

    token = get_chunkedgraph_secret()
    session.headers['Authorization'] = f"Bearer {token}"

    futures = {}
    url = 'https://prod.flywire-daf.com/neurons/api/v1/mark_completion'
    for i_, x_, y_, z_, u_ in zip(x.valid_id.values,
                                 x.x.values,
                                 x.y.values,
                                 x.z.values,
                                 x.user_id.values):
        post = dict(valid_id=str(i_),
                    location=f'{x_}, {y_}, {z_}',
                    action='single',
                    user_id=u_)

        f = future_session.post(url, data=post)
        futures[f] = post

    # Get the responses
    resp = [f.result() for f in navis.config.tqdm(futures,
                                                  desc='Submitting',
                                                  disable=not progress or len(futures) == 1,
                                                  leave=False)]

    success = []
    errors = []
    for r, f in zip(resp, futures):
        try:
            r.raise_for_status()
        except BaseException as e:
            success.append(False)
            errors.append(str(e))
            continue

        if 'Success' not in r.text:
            success.append(False)
            errors.append(r.text)
            continue

        success.append(True)
        errors.append(None)

    x = x[['valid_id', 'x', 'y', 'z']].copy()
    x['success'] = success
    x['errors'] = errors

    if not all(x.success):
        navis.config.logger.error(f'Encountered {(~x.success).sum()} errors '
                                  'while submitting cell identifications. Please '
                                  'see the `errors` columns in the returned '
                                  'dataframe for details.')
    else:
        navis.config.logger.info('SUCCESS!')

    return x
