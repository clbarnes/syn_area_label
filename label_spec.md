# Label file specification

- Postsynaptic sites whose ROIs nearby are included in the same HDF5 file
- The presynaptic site is a new node close to the connector node
  - The mappings from presynaptic site ID to connector node ID is found in `/annotations/presynaptic_site/pre_to_conn`
- The postsynaptic site is the treenode postsynaptic to the connector node
- Partners are one to one
- Clefts are labelled with a 1px brush but may still need skeletonising
- Image data is lossless from N5 export
- File names are `"data_{n}.hdf5"`

## to do

- Information about the contained edges are serialised in pytables format in `/tables/connectors`
- Areas are stored in pytables-serialised dataframes in `table.hdf5`
  - `connectors` group contains edge information, including area
  - `skeletons` group contains several tables sharing an index:
    - `skeletons` group contains ID, name, side, segment
    - `classes` group contains one-hot encoded neuron classes
    - `superclasses` group contains one-hot encoded superclasses
    - `annotations` group contains one-hot encoded annotations
