run_3d_leapfrog:
    uv run python 3D/initialize.py --init_cond leapfrog --dir output_3d_leapfrog
    uv run python 3D/advance.py --init_cond leapfrog --dir output_3d_leapfrog > output_3d_leapfrog/log.txt
