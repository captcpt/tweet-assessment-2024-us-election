import pandas as pd
import geopandas as gpd
import folium
import ast
import json

# Load your labeled election data (from Twitter predictions)
df = pd.read_csv('./final_data/labeled_election_data.csv')
df = df[df['prediction'].isin([0, 1])]

# Calculate the percentage of prediction 0 and 1 per state
state_prediction_pct = df.groupby('state')['prediction'].value_counts(normalize=True).unstack(fill_value=0)
state_prediction_pct['percent_0'] = state_prediction_pct[0] * 100
state_prediction_pct['percent_1'] = state_prediction_pct[1] * 100

# Process hashtags
df['hashtags_list'] = df['hashtags_list'].apply(lambda x: ast.literal_eval(x) if x != '[]' else [])
df_filtered = df[df['hashtags_list'].map(len) > 0]

hashtags_expanded = (
    df_filtered.explode("hashtags_list")
    .dropna(subset=["hashtags_list"])
    .groupby(["state", "hashtags_list"])
    .size()
    .reset_index(name="count")
)

top_hashtags = (
    hashtags_expanded.groupby("state")
    .apply(lambda x: x.nlargest(5, "count"))
    .reset_index(drop=True)
)

top_hashtags['formatted'] = top_hashtags.groupby('state')['hashtags_list'].transform(
    lambda x: '<br>'.join([f"{i+1}. {hashtag}" for i, hashtag in enumerate(x)])
)

# Load the actual results from the JSON file
# Taken from Reuters + Normalized: https://www.reuters.com/graphics/USA-ELECTION/RESULTS/zjpqnemxwvx/
with open('./final_data/actual_results.json') as f:
    actual_results = json.load(f)

# Convert the actual results into a DataFrame
actual_df = pd.DataFrame(actual_results)

# Merge the prediction data with the actual election results by state
merged_df = state_prediction_pct.merge(actual_df, how='left', on='state')
merged_df['Dem_diff'] = merged_df['percent_0'] - merged_df['Dem']
merged_df['Rep_diff'] = merged_df['percent_1'] - merged_df['Rep']

merged_df.head()

# Load and prepare geodata
gdf = gpd.read_file('./state_shape/cb_2018_us_state_500k/cb_2018_us_state_500k.shp')
gdf = gdf.to_crs("EPSG:4326")

# Merge all data
gdf = gdf.merge(state_prediction_pct, how="right", left_on="NAME", right_on="state")
gdf = gdf.merge(merged_df[['state', 'Dem', 'Rep']], how="left", left_on="NAME", right_on="state")
top_hashtags_merged = top_hashtags.groupby("state")["formatted"].first().reset_index()
gdf = gdf.merge(top_hashtags_merged, how="left", left_on="NAME", right_on="state")
gdf = gdf.dropna(subset=['geometry'])


# Create base map
m = folium.Map(location=[37.8, -96], zoom_start=4, tiles="cartodbpositron")

# Create predicted layer
predicted_layer = folium.FeatureGroup(name="Predicted Results")
for _, row in gdf.iterrows():
    popup_text = f"""
    <h4>{row['NAME']}</h4>
    <b>Predicted Democrats:</b> {row['percent_0']:.2f}%<br>
    <b>Predicted Republicans:</b> {row['percent_1']:.2f}%<br>
    <b>Top 5 Hashtags:</b><br>{row['formatted'] if pd.notnull(row['formatted']) else "No hashtags available"}
    """
    color = "blue" if row['percent_0'] > row['percent_1'] else "red"
    folium.GeoJson(
        row.geometry,
        style_function=lambda x, color=color: {"fillColor": color, "color": "black", "weight": 0.5, "fillOpacity": 0.6},
        tooltip=folium.Tooltip(popup_text),
    ).add_to(predicted_layer)

# Create actual results layer
actual_layer = folium.FeatureGroup(name="Actual Results")
for _, row in gdf.iterrows():
    popup_text = f"""
    <h4>{row['NAME']}</h4>
    <b>Actual Democrats:</b> {row['Dem']:.2f}%<br>
    <b>Actual Republicans:</b> {row['Rep']:.2f}%<br>
    <b>Top 5 Hashtags:</b><br>{row['formatted'] if pd.notnull(row['formatted']) else "No hashtags available"}
    """
    color = "blue" if row['Dem'] > row['Rep'] else "red"
    folium.GeoJson(
        row.geometry,
        style_function=lambda x, color=color: {"fillColor": color, "color": "black", "weight": 0.5, "fillOpacity": 0.6},
        tooltip=folium.Tooltip(popup_text),
    ).add_to(actual_layer)

# Add layers and controls
predicted_layer.add_to(m)
actual_layer.add_to(m)
folium.LayerControl().add_to(m)

# Save map
m.save("state_predictions_vs_actual_map.html")