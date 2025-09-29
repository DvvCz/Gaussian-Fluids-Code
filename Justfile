train_3d_leapfrog:
    uv run python 3D/initialize.py --init_cond leapfrog --dir output_3d_leapfrog

run_3d_leapfrog start_frame="0" last_time="2":
    uv run python 3D/advance.py --init_cond leapfrog --dt 0.1 --start_frame {{start_frame}} --last_time {{last_time}} --dir output_3d_leapfrog > output_3d_leapfrog/log.txt
