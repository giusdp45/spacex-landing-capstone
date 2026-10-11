# SpaceX Falcon 9 Landing Capstone

Repository: https://github.com/giusdp45/spacex-landing-capstone

This original project investigates historical Falcon 9 landing outcomes using API collection code, web scraping, data wrangling, SQL, visual EDA, Folium, Dash and four classifiers. The PDF presentation explains methods, results and limitations.

## Run

```bash
python -m venv .venv
pip install -r requirements.txt
python scripts/analyze.py
python scripts/build_maps.py
python scripts/spacex_dash_app.py
```

Open http://127.0.0.1:8051 for Dash. Open maps/*.html for the Folium maps; internet access supplies OpenStreetMap tiles. Jupyter notebooks in notebooks/ contain complete code and executed results. Open notebooks from that directory so their relative paths resolve correctly.

## Data and provenance

- The official IBM course frozen dataset contains 90 single-payload Falcon 9 launches from 2010-06-04 through 2020-11-05. It is used for visual EDA and prediction.
- Wikipedia revision 1027686922 supplies 122 scraped Falcon 9 records through 2021-06-06 for SQL. It is a separate sample; the SQL answers reflect that cutoff.
- The official Dash and geospatial samples contain 56 launches and four historical site labels (2010-2018).
- OpenStreetMap geometry is fetched from its map API for bbox -80.61,28.55,-80.56,28.59. Proximity measures refer to mapped segments within that bounding box. Rail includes onsite transporter rails, and roads include service roads.

The live SpaceX API returned HTTP 525 during collection. scripts/collect_api.py and the executed collection notebook preserve the request and explain the official snapshot fallback. We do not claim to have collected live API results successfully.

Sources:

- https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv
- https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv
- https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv
- https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv
- https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922
- https://github.com/r-spacex/SpaceX-API/tree/master/docs/v4
- https://www.openstreetmap.org/copyright

## Main findings

The 90-row dataset has 60 successful landing labels (66.7%). KSC LC 39A has 17/22 class-1 launches; VAFB SLC 4E 10/13; CCAFS SLC 40 33/55. SVM leads five-fold training cross-validation at 88.9%, with 77.8% held-out accuracy and 66.7% balanced accuracy (18 launches). It predicts all 12 successes correctly but incorrectly predicts success for four of six class-0 launches. A chronological evaluation selects a decision tree with 88.9% test accuracy, against an 83.3% majority baseline.

Class 0 includes no attempt and unknown outcomes; Class 1 includes ocean landings. These labels do not establish recovery for reuse or monetary savings. Site, orbit and payload associations are confounded by mission type and era. A larger, recent dataset, attempted-landing target and forward validation are needed before operational decisions.
