"""Fig.4-style token visualization on the ScanNet GT mesh.

The compression methods pick 3D regions; this package paints those regions
back onto ``<scene>_vh_clean_2.ply`` so a figure shows the scene's own colour
where a method kept evidence, and gray where it did not.

Modules
-------
scannet_mesh          read/write ScanNet PLY meshes, axis alignment
voxel_mapping         the forward pass' voxel rule, mesh voxel hash map
token_selection       the per-run selection record (.npz) and its loaders
export_token_selection forward-time recorder used by the video tower
render_mesh_selection colours, PLY export, offscreen PNG rendering
visualize_fig4        entry point
"""
