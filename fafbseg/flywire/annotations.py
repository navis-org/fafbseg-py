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
import pytz
import warnings

import datetime as dt
import numpy as np
import pandas as pd

from requests_futures.sessions import FuturesSession

from ..utils import make_iterable
from .utils import get_cave_client, retry, get_chunkedgraph_secret
from .segmentation import (locs_to_segments, supervoxels_to_roots, is_latest_root,
                           update_ids)


__all__ = ['get_somas', 'get_materialization_versions',
           'create_annotation_table', 'get_annotation_tables',
           'get_annotation_table_info', 'get_annotations',
           'delete_annotations', 'upload_annotations',
           'is_proofread', 'find_celltypes', 'list_annotation_tables']


PR_TABLE = None
PR_META = None
ANNOTATION_TABLE = "neuron_information_v2"
_annotation_table = None


def is_proofread(x, cache=True, validate=True):
    """Test if neuron has been set to `proofread`.

    Under the hood, this uses a cached version of the proofreading table which
    is updated everytime this function gets called.

    Parameters
    ----------
    x  :            int | list of int
                    Root IDs to check.
    validate :      bool
                    Whether to validate IDs.
    cache :         bool
                    Use and update a locally cached version of the proofreading
                    table. Setting this to ``False`` will force fetching the
                    full table which is considerably slower.

    Returns
    -------
    proofread :     np.ndarray
                    Boolean array.

    """
    global PR_TABLE, PR_META

    if not isinstance(x, (np.ndarray, set, list, pd.Series)):
        x = [x]

    # Force into array and convert to integer
    x = np.asarray(x, type=np.int64)

    # Check if any of the roots are outdated -> can't check those
    if validate:
        il = is_latest_root(x)
        if any(~il):
            print("At least some root ID(s) outdated and will therefore show up as "
                  f"not proofread: {x[~il]}")

    # Get available materialization versions
    client = get_cave_client('production')
    mat_versions = client.materialize.get_versions()

    # Check if the cached version is outdated
    if cache and PR_META:
        if max(mat_versions) > PR_META['mat_version']:
            PR_TABLE = None
            PR_META = None

    # If nothing cached catch the table from scratch
    if isinstance(PR_TABLE, type(None)) or not cache:
        # This ought to automatically use the most recent materialization version
        PR_META = dict(timestamp=dt.datetime.utcnow(),
                       mat_version=max(mat_versions))
        PR_TABLE = client.materialize.live_query(table='proofreading_status_public_v1',
                                                 timestamp=PR_META['timestamp'])
        # Only keep relevant rows and columns
        PR_TABLE = PR_TABLE.loc[PR_TABLE.valid == 't', ['pt_supervoxel_id', 'pt_root_id']]
    else:
        # Update root IDs
        now = dt.datetime.utcnow()
        # get_delta_roots currently acts up if there are no new roots
        # (i.e. if this gets called in quick succession)
        try:
            old_roots, _ = client.chunkedgraph.get_delta_roots(PR_META['timestamp'], now)
        except requests.exceptions.HTTPError as e:
            if "need at least one array" in str(e):
                old_roots = []
            else:
                raise
        except BaseException:
            raise

        to_update = PR_TABLE.pt_root_id.isin(old_roots)
        if any(to_update):
            new_roots = supervoxels_to_roots(PR_TABLE.loc[to_update, 'pt_supervoxel_id'].values)
            PR_TABLE.loc[to_update, 'pt_root_id'] = new_roots
        PR_META['timestamp'] = now

    return np.isin(x, PR_TABLE.pt_root_id.values)


def get_materialization_versions(dataset='production'):
    """Fetch info on the available materializations."""
    # Get/Initialize the CAVE client
    client = get_cave_client(dataset)

    # Get currently existing versions
    versions = client.materialize.get_versions()

    # Fetch meta data
    meta = pd.DataFrame.from_records([client.materialize.get_version_metadata(v) for v in versions])

    # Reduce precision for timestamps to make more readable
    meta['expires_on'] = meta.expires_on.values.astype('datetime64[m]')
    meta['time_stamp'] = meta.time_stamp.values.astype('datetime64[m]')

    return meta.sort_values('version', ascending=False).reset_index(drop=True)


def create_annotation_table(name: str,
                            schema: str,
                            description: str,
                            voxel_resolution=[1, 1, 1],
                            dataset='production'):
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

    Returns
    -------
    response
                        Server response if something went wrong.

    """
    # Get/Initialize the CAVE client
    client = get_cave_client(dataset)

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


def list_annotation_tables(dataset='production'):
    """Fetch available annotation tables."""
    # Get/Initialize the CAVE client
    client = get_cave_client(dataset)

    an = client.annotation.get_tables()
    ma = client.materialize.get_tables()

    all_tables = list(set(an) | set(ma))

    df = pd.DataFrame(index=all_tables)
    df['annotation'] = df.index.isin(an)
    df['materialized'] = df.index.isin(ma)

    return df


def get_annotation_tables(dataset='production'):
    """Fetch available annotation tables."""
    warnings.warn(
            "`get_annotation_tables` is deprecated and will be removed in a "
            "future version of fafbseg, please use `list_annotation_tables`"
            "instead",
            DeprecationWarning,
            stacklevel=2
        )
    return list_annotation_tables(dataset=dataset)


def get_annotation_table_info(table_name: str,
                              dataset='production'):
    """Get info for given table.

    Parameters
    ----------
    table_name :        str
                        Name of the table.

    Returns
    -------
    info :              dict

    """
    # Get/Initialize the CAVE client
    client = get_cave_client(dataset)

    return client.annotation.get_table_metadata(table_name)


def get_annotations(table_name: str,
                    materialization='live',
                    split_positions: bool = False,
                    drop_invalid: bool = True,
                    dataset: str = 'production',
                    **filters):
    """Get annotations from given table.

    Parameters
    ----------
    table_name :        str
                        Name of the table.
    materialization :   "live" | "latest" | int | bool
                        Which materialization version to fetch. You can also
                        provide an ID (int) for a specific materialization
                        version (see ``get_materialization_versions``). Set to
                        False to fetch the non-materialized version.
    split_positions :   bool
                        Whether to split x/y/z positions into separate columns.
    drop_invalid :      bool
                        Whether to drop invalid (i.e. deleted or updated)
                        annotations.
    **filters
                        Additional filter queries. See Examples. This works only
                        if ``materialization!=False``.

    Returns
    -------
    table :             pandas.DataFrame

    """
    # Get/Initialize the CAVE client
    client = get_cave_client(dataset)

    navis.utils.eval_param(table_name, name='table_name', allowed_types=(str, ))

    if materialization == 'live':
        data = client.materialize.live_query(table=table_name,
                                             timestamp=dt.datetime.utcnow(),
                                             split_positions=split_positions,
                                             **filters)
    elif materialization:
        if materialization == 'latest':
            materialization = get_materialization_versions(dataset=dataset).version.max()

        data = client.materialize.query_table(
                       materialization_version=materialization,
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


def delete_annotations(table_name: str,
                       annotation_ids: list,
                       dataset='production'):
    """Delete annotations from table.

    Parameters
    ----------
    table_name :        str
                        Name of the table.
    annotation_ids :    int | list | np.ndarray | pandas.DataFrame
                        ID(s) of annotations to delete. If DataFrame must contain
                        an "id" column.

    Returns
    -------
    response :          str
                        Server response.

    """
    # Get/Initialize the CAVE client
    client = get_cave_client(dataset)

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


def upload_annotations(table_name: str,
                       data: pd.DataFrame,
                       dataset='production'):
    """Upload or update annotations to table.

    Parameters
    ----------
    table_name :        str
                        Name of the table.
    data :              pandas.DataFrame
                        Data to be uploaded. Must match the table's schema! If
                        'id' column exists, we assume that you want to update
                        existing annotations (i.e. rows in the table) with the
                        given IDs. See Examples for details.

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
    client = get_cave_client(dataset)

    navis.utils.eval_param(table_name, name='table_name', allowed_types=(str, ))
    navis.utils.eval_param(data, name='data', allowed_types=(pd.DataFrame, ))

    if 'id' in data.columns:
        func = client.annotation.update_annotation_df
    else:
        func = client.annotation.post_annotation_df

    resp = func(table_name=table_name, df=data, position_columns=None)

    print('Success! See response for details.')

    return resp


def get_somas(x=None, materialization='live', split_positions=False, dataset='production'):
    """Fetch nuclei segmentation for given neuron(s).

    Parameters
    ----------
    x :                 int | list of ints | NeuronList, optional
                        FlyWire root ID(s) or neurons for which to fetch soma
                        infos. Use ``None`` to fetch complete list of annotated
                        nuclei. If neurons, will set their soma and soma radius
                        if one is found.
    materialization :   "live" | "latest" | int | bool
                        Which materialization version to fetch. You can also
                        provide an ID (int) for a specific materialization
                        version (see ``get_materialization_versions``). Set to
                        False to fetch the non-materialized version.
    split_positions :   bool
                        Whether to have separate columns for x/y/z position.

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
        filter_in_dict = {'pt_root_id': root_ids}

    nuc = get_annotations('nuclei_v1',
                          materialization=materialization,
                          split_positions=split_positions,
                          dataset=dataset,
                          filter_in_dict=filter_in_dict)

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

        # If NeuronList, set their somas
        if isinstance(x, navis.NeuronList):
            soma_pos = nuc.set_index('pt_root_id').pt_position.to_dict()
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

    Use this bulk submission of cell identification with great care!

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
        existing = _get_cell_type_table(update_ids=True)

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

        if not 'Success' in r.text:
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


def find_celltypes(x,
                   user=None,
                   exact=False,
                   case=False,
                   regex=True,
                   update_roots=True):
    """Search cell identification annotations for given term/root IDs.

    Parameter
    ---------
    x :         str | int | Neuron/List | list of ints
                Term (str) or root ID(s) to search for.
    user :      id | list thereof, optional
                If provided will only return annotations from given user(s).
                Currently requires user ID.
    exact :     bool
                Whether term must be an exact match. For example, if
                ``exact=False`` (default), 'sensory' will match to e.g.
                'sensory,olfactory' or 'AN sensory'.
    case :      bool
                If True (default = False), search for term will be case
                sensitive.
    regex :     bool
                Whether to interpret term as regex.
    update_roots : bool
                Whether to update root IDs for matches.

    Returns
    -------
    pandas.DataFrame
                If `x` is root IDs
                Pandas DataFrame with annotations matching ``x``. Coordinates
                are in 4x4x40nm voxel space.

    """
    # See if ``x`` is a root ID as string
    if isinstance(x, str):
        try:
            x = np.int64(x)
        except ValueError:
            pass

    if isinstance(x, (int, np.int64, np.int32)):
        x = np.array([x])
    elif isinstance(x, navis.NeuronList):
        x = x.id
    elif isinstance(x, navis.BaseNeuron):
        x = np.array([x])

    # Check if root IDs are outdated
    if isinstance(x, (tuple, pd.Series, list, set, np.ndarray)):
        x = np.asarray(x)
        is_latest = is_latest_root(x)
        if any(~is_latest):
            raise ValueError(f'Some root IDs are outdated: {x[~is_latest]}')

    # We only need to update root IDs if we're
    # looking for annotations for given cells
    ct = _get_cell_type_table(update_ids=not isinstance(x, str))

    # If requested, restrict to given user
    if not isinstance(user, type(None)):
        if isinstance(user, (str, int)):
            ct = ct[ct.user_id == user]
        else:
            ct = ct[ct.user_id.isin(user)]

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

        if update_roots and len(ct):
            new_roots = update_ids(ct.pt_root_id.values,
                                   supervoxels=ct.pt_supervoxel_id.values)
            if any(new_roots.changed.values):
                ct['pt_root_id'] = ct.pt_root_id.map(new_roots.set_index('old_id').new_id.to_dict())

        # Rename columns to make it less clunky to work with
        ct = ct.rename({'pt_position_x': 'pos_x',
                        'pt_position_y': 'pos_y',
                        'pt_position_z': 'pos_z',
                        'pt_root_id': 'root_id',
                        'pt_supervoxel_id': 'supervoxel_id'},
                        axis=1)

        # Convert from nm to voxel space
        ct[['pos_x', 'pos_y', 'pos_z']] /= [4, 4, 40]
    else:
        ct = ct[ct.pt_root_id.isin(x)]

    return ct


def _get_cell_type_table(force_new=False, update_ids=False):
    """Fetch (and cache) annotation table."""
    global _annotation_table

    client = get_cave_client()

    # Check what the latest materialization version is
    mds = client.materialize.get_versions_metadata()
    mds = sorted(mds, key=lambda x: x['time_stamp'])
    mat_version = mds[-1]['version']

    if not force_new:
        # Check if table needs to be voided
        if not isinstance(_annotation_table, type(None)):
            if _annotation_table.attrs['mat_version'] < mat_version:
                force_new = True

    if isinstance(_annotation_table, type(None)) or force_new:
        # If no table, fetch from scratch
        now = pytz.UTC.localize(dt.datetime.utcnow())
        _annotation_table = get_annotations(ANNOTATION_TABLE, split_positions=True)
        _annotation_table.attrs['time_fetched'] = now
        _annotation_table.attrs['time_updated'] = now
        _annotation_table.attrs['mat_version'] = mat_version

    if update_ids:
        now = pytz.UTC.localize(dt.datetime.utcnow())
        expired, new = client.chunkedgraph.get_delta_roots(
                         timestamp_past=_annotation_table.attrs['time_updated'],
                         timestamp_future=now
                         )
        needs_update = np.isin(_annotation_table.pt_root_id, expired)
        _annotation_table.loc[needs_update,
                              'pt_root_id'] = supervoxels_to_roots(_annotation_table.loc[needs_update, 'pt_supervoxel_id'])

    return _annotation_table


def mark_cell_completion(x, validate=True, skip_existing=True,
                         max_threads=4, progress=True):
    """Submit proofread status for given cell.

    Use this bulk submission of proofreading status with great care!

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
            navis.config.logger.info(f'Looks like all neurons have already '
                                     'been set to proofread')
            return pd.DataFrame(columns=['valid_id', 'x', 'y', 'z', 'success', 'errors'])

    session = requests.Session()
    future_session = FuturesSession(session=session, max_workers=max_threads)

    token = get_chunkedgraph_secret()
    session.headers['Authorization'] = f"Bearer {token}"

    futures = {}
    url = f'https://prod.flywire-daf.com/neurons/api/v1/mark_completion'
    for i, (_, row) in enumerate(x.iterrows()):
        post = dict(valid_id=str(row.valid_id),
                    location=f'{row.x}, {row.y}, {row.z}',
                    action='single',
                    user_id=row.get('user_id', ''))

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
            sucess.append(False)
            errors.append(str(e))
            continue

        if not 'Success' in r.text:
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
