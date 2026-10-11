"""Build Folium maps of launch sites, outcomes and measured geographic proximity."""
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import folium
from folium.plugins import MarkerCluster, MousePosition
from geopy.distance import geodesic

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MAPS = ROOT / "maps"
MAPS.mkdir(exist_ok=True)


def closest_on_polyline(origin, points):
    scale_x = 111320 * math.cos(math.radians(origin[0]))
    scale_y = 111320
    def xy(point):
        return np.array([(point[1] - origin[1]) * scale_x, (point[0] - origin[0]) * scale_y])
    best = None
    for a, b in zip(points[:-1], points[1:]):
        av, bv = xy(a), xy(b)
        delta = bv - av
        t = np.clip(-av.dot(delta) / max(delta.dot(delta), 1e-12), 0, 1)
        v = av + t * delta
        point = (origin[0] + v[1] / scale_y, origin[1] + v[0] / scale_x)
        distance = geodesic(origin, point).km
        if best is None or distance < best[0]:
            best = (distance, point)
    return best


def build_maps():
    df = pd.read_csv(DATA / "spacex_launch_geo.csv")
    sites = df.groupby("Launch Site")[["Lat", "Long"]].first().reset_index()
    site_map = folium.Map(location=[36, -96], zoom_start=4, tiles="OpenStreetMap")
    for row in sites.itertuples(index=False, name=None):
        name, lat, lon = row
        folium.CircleMarker([lat, lon], radius=10, color="#1d4ed8", fill=True, tooltip=name).add_to(site_map)
        folium.Marker([lat, lon], popup=f"{name}<br>{lat:.6f}, {lon:.6f}", tooltip=name).add_to(site_map)
    site_map.save(MAPS / "launch_sites.html")
    records_map = folium.Map(location=[28.56, -80.58], zoom_start=9)
    cluster = MarkerCluster(name="56 launch outcomes").add_to(records_map)
    for _, row in df.iterrows():
        color = "green" if row["class"] else "red"
        folium.Marker([row.Lat, row.Long], icon=folium.Icon(color=color, icon="info-sign"),
                      popup=f"Flight {row['Flight Number']} | {row.Date}<br>{row['Launch Site']}<br>Class {row['class']}",
                      tooltip=f"{row['Launch Site']} - class {row['class']}").add_to(cluster)
    folium.LayerControl().add_to(records_map)
    records_map.save(MAPS / "launch_records.html")
    origin = tuple(sites.loc[sites["Launch Site"] == "CCAFS SLC-40", ["Lat", "Long"]].iloc[0])
    proximity_map = folium.Map(location=origin, zoom_start=13)
    folium.Marker(origin, tooltip="CCAFS SLC-40", popup="Launch site", icon=folium.Icon(color="blue")).add_to(proximity_map)
    osm_file = DATA / "osm_proximity.json"
    if not osm_file.exists():
        response = requests.get("https://api.openstreetmap.org/api/0.6/map", params={"bbox": "-80.61,28.55,-80.56,28.59"}, timeout=60)
        response.raise_for_status()
        tree = ET.fromstring(response.text)
        nodes = {n.attrib['id']: {'lat': float(n.attrib['lat']), 'lon': float(n.attrib['lon'])} for n in tree.findall('node')}
        elements = []
        for way in tree.findall('way'):
            tags = {t.attrib['k']: t.attrib['v'] for t in way.findall('tag')}
            if any(t in tags for t in ['natural', 'railway', 'highway']):
                elements.append({'id': int(way.attrib['id']), 'tags': tags, 'geometry': [nodes[n.attrib['ref']] for n in way.findall('nd') if n.attrib['ref'] in nodes]})
        osm_file.write_text(json.dumps({'source': 'https://api.openstreetmap.org/api/0.6/map?bbox=-80.61,28.55,-80.56,28.59', 'elements': elements}), encoding="utf-8")
    elements = json.loads(osm_file.read_text(encoding="utf-8"))["elements"]
    categories = {"Coastline": ("natural", "coastline", "#0891b2"), "Railway": ("railway", "rail", "#9333ea"), "Road": ("highway", None, "#d97706")}
    measured = []
    for category, (tag, value, color) in categories.items():
        candidates = []
        for element in elements:
            tags = element.get("tags", {})
            if tag not in tags or (value is not None and tags[tag] != value):
                continue
            points = [(p["lat"], p["lon"]) for p in element.get("geometry", [])]
            if len(points) < 2:
                continue
            result = closest_on_polyline(origin, points)
            candidates.append((result[0], result[1], element["id"], tags.get("name", category)))
            folium.PolyLine(points, color=color, weight=2, opacity=.45, tooltip=tags.get("name", category)).add_to(proximity_map)
        if candidates:
            distance, point, way_id, name = min(candidates)
            folium.PolyLine([origin, point], color=color, weight=4, tooltip=f"{category}: {distance:.3f} km").add_to(proximity_map)
            folium.Marker(point, tooltip=f"Nearest {category}: {distance:.3f} km", popup=f"OSM way {way_id}: {name}").add_to(proximity_map)
            measured.append({"Feature": category, "DistanceKm": distance, "OSMWay": way_id, "Name": name, "Lat": point[0], "Long": point[1]})
    MousePosition().add_to(proximity_map)
    proximity_map.save(MAPS / "proximity_map.html")
    result = {"rows": len(df), "successes": int(df['class'].sum()), "sites": sites.to_dict(orient="records"), "proximity": measured,
              "site_rates": df.groupby('Launch Site')['class'].agg(['sum', 'size', 'mean']).reset_index().to_dict(orient='records'),
              "proximity_method": "Closest projection onto OSM way segments in bbox -80.61,28.55,-80.56,28.59, then WGS84 geodesic distance; approximate and map-dependent."}
    (DATA / "map_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(build_maps(), indent=2))
