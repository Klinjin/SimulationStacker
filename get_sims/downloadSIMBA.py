import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
# This script downloads files from specified URLs using wget and saves them to designated directories.

def download_file(url, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "wget",
        "-nd",
        "-nc",
        "-e", "robots=off",
        "-r",
        "-l", "1",
        "-P", output_dir,
        url
    ]

    print(f"Starting download: {url} -> {output_dir}")
    try:
        subprocess.run(cmd, check=True)
        print(f"Finished: {url}")
    except subprocess.CalledProcessError as e:
        print(f"Failed: {url} with error: {e}")

def parallel_download(urls, output_dirs, max_workers=4):
    if len(urls) != len(output_dirs):
        raise ValueError("URLs and output directories must be the same length.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_file, url, outdir)
            for url, outdir in zip(urls, output_dirs)
        ]
        for future in as_completed(futures):
            # This raises exceptions if any occurred
            future.result()


# Example usage
if __name__ == "__main__":
    snapshot = 136
    urls = [
        # f"http://simba.roe.ac.uk/simdata/m50n512/s50/snapshots/snap_m50n512_{snapshot}.hdf5",
        # f"http://simba.roe.ac.uk/simdata/m50n512/s50noagn/snapshots/snap_m50n512_{snapshot}.hdf5",
        # f"http://simba.roe.ac.uk/simdata/m50n512/s50nox/snapshots/snap_m50n512_{snapshot}.hdf5",
        # f"http://simba.roe.ac.uk/simdata/m50n512/s50nojet/snapshots/snap_m50n512_{snapshot}.hdf5",
        # f"http://simba.roe.ac.uk/simdata/m50n512/s50nofb/snapshots/snap_m50n512_{snapshot}.hdf5",
        f"http://simba.roe.ac.uk/simdata/m100n1024/s50/snapshots/snap_m100n1024_{snapshot}.hdf5"
    ]
    output_dirs = [
        # "/pscratch/sd/r/rhliu/simulations/SIMBA/m50n512/s50/snapshots/",
        # "/pscratch/sd/r/rhliu/simulations/SIMBA/m50n512/s50noagn/snapshots/",
        # "/pscratch/sd/r/rhliu/simulations/SIMBA/m50n512/s50nox/snapshots/",
        # "/pscratch/sd/r/rhliu/simulations/SIMBA/m50n512/s50nojet/snapshots/",
        # "/pscratch/sd/r/rhliu/simulations/SIMBA/m50n512/s50nofb/snapshots/",
        "/pscratch/sd/r/rhliu/simulations/SIMBA/m100n1024/s50/snapshots/"
    ]
    parallel_download(urls, output_dirs, max_workers=5)