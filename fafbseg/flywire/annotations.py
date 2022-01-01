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

import datetime as dt
import numpy as np
import pandas as pd

from .utils import get_cave_client, retry
from .segmentation import locs_to_segments


__all__ = ['get_somas', 'get_materialization_versions',
           'create_annotation_table', 'get_annotation_tables',
           'get_annotation_table_info', 'get_annotations',
           'delete_annotations', 'upload_annotations']


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


def get_annotation_tables(dataset='production'):
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
                    split_positions=False,
                    drop_invalid: bool = True,
                    dataset='production',
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
            root_ids = x.id.astype(int)
        else:
            root_ids = navis.utils.make_iterable(x).astype(int)
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
                if int(n.id) not in soma_pos:
                    continue

                if isinstance(n, navis.TreeNeuron):
                    n.soma = n.snap(soma_pos[int(n.id)])[0]
                    n.nodes.loc[n.nodes.node_id == n.soma, 'radius'] = soma_rad[int(n.id)]
                    n._clear_temp_attr()  # not sure why but this is necessary for some reason
                    n.reroot(n.soma, inplace=True)
                elif isinstance(n, navis.MeshNeuron):
                    n.soma_pos = soma_pos[int(n.id)]
    else:
        # Make sure the column exist even in an empty table
        nuc['rad_est'] = []

    # Sorting by radius makes sure that the small false-positives end up at the
    # bottom of the list
    return nuc.sort_values('rad_est', ascending=False)
