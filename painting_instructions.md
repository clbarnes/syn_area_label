# Instructions for cleft labelling

[BigCAT UI docs here](https://github.com/saalfeldlab/bigcat/wiki/BigCat-User-Interface)

## Installation

- Install [bigcat](https://github.com/saalfeldlab/bigcat)

## Opening file

Be careful to keep the Z location aligned with pixels,
and not to rotate the view.

- Open with `bigcat -i data_X.hdf5 -r /volumes/raw -l /volumes/labels/clefts`
- Hold space and scroll the mouse wheel to get a 1px brush
  - Label as close to 1px thick as possible
  - thicker lines will be skeletonised
- Press `a` to open annotations; select one and press `g` to go to it.

## Painting

- Select a presynaptic annotation and press `g` to go to it
- Press `n` to get a new ID
- Hold space and left click to paint
  - Space and right click to erase
- Label the postsynaptic membrane in all slices where PSDs are visible
- Ctrl+C to comment on the presynaptic annotation with the label ID
- Go to a new annotation, get a new ID, etc.

`s`, `Ctrl+s`, `Ctrl+Shift+s` to save everything
