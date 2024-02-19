# Instructions for cleft labelling

[BigCAT UI docs here](https://github.com/saalfeldlab/bigcat/wiki/BigCat-User-Interface)

> WARNING
>
> There may be a bug in bigCAT which prevents saving comments,
> which were originally used to map the IDs of the painted labels with the edge IDs.
>
> Details can be found here: <https://github.com/saalfeldlab/bigcat/issues/117>
>
> You can still save the painted image volumes without issue.

## Installation

- Install [bigcat](https://github.com/saalfeldlab/bigcat)

## Opening file

Be careful to keep the Z location aligned with pixels,
and not to rotate the view.

- Open with `bigcat -i path/to/my.hdf5 -r /volumes/raw -l /volumes/labels/clefts`
  - `scripts/open_cremi.sh path/to/my.hdf5` is equivalent
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
- Ctrl+C to comment on the presynaptic annotation with the ID of the label you've just used
  - Make sure you have the bigCAT window selected when you Ctrl+c, NOT the terminal which is running it!
  - Due to a possible bug (see warning above), the mapping between edge ID and label ID should also be tracked external to bigCAT (TSV etc.)
- Go to a new annotation, get a new ID, etc.

`s`, `Ctrl+s`, `Ctrl+Shift+s` save different combinations of things.

- `Ctrl+s` saves the painting canvas
- `s` is required to save the comments (see warning above)

## Escaping off-axis views

Do not:

- Rotate the view (click and drag)
- Scroll through Z in non-integer multiples of the Z resolution (`Ctrl` or `Shift` + scroll)

You can re-align the Z location with the slicing plane by `g`oing any of the annotations.

You can reset the orientation to align with the axis with `Shift + z`.
