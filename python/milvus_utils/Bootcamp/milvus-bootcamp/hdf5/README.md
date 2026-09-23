# Running create hdf5 file

## Prerequisites
* Python >= 3.10
* h5py and pandas python packages 

### Installation notes
create_hdf5.py requires h5py and pandas python packages installed with virtual python environment:

```shell
python3 -m venv myenv
source myenv/bin/activate
pip3 install h5py
pip3 install pandas
```

### Run
python3 create_hdf5.py --base=<fbin base file> --query=<fbin query file> --gt=<fbin/parquet groundtruth file> --hdf5=<output hdf5 file>

