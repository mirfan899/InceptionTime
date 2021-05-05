## Install requirements
I have tested the code on Python 3.6.9
```shell
pip install -r requirements.txt
```

## Directory structure
Create and place the csv files in data directory.

Create a features directory

## Train the model.
run the script 
```shell
ts.py
```

## Run API flask server
```shell
python api.py
```

Test it by providing values as json. You need to provide only `cp`,`kvFst`,`kvTrgger`,`ripple_angle` and `wave_angle` values.
```shell
curl -X 'POST' 'http://127.0.0.1:5000/api/wave_type' -H 'Content-Type: application/json' -d '{"row": "1111,-1.81898940354586E-12,-2.59855629077979E-13,41.1859251657097,41.1859251657097"}'
```