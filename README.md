**New Orleans Traffic Accidents**

> An investigation of the causes of traffic accidents in New Orleans, LA.

**Contents**

- [Project Proposal](#project-proposal)
    - [The Problem](#the-problem)
    - [Potential Clients](#potential-clients)
    - [Data](#data)
    - [Problem Solving Approach](#problem-solving-approach)
    - [Deliverables](#deliverables)
- [Methods](#methods)
    - [Inital Investigation](#inital-investigation)
        - [Codes](#codes)
    - [Data Wrangling](#data-wrangling)
- [Results](#results)
    - [Statistics/Visuals](#statisticsvisuals)
        - [Maps](#maps)
        - [Charts](#charts)
        - [Accident Counts by Date](#accident-counts-by-date)
        - [Weather Distribution](#weather-distribution)
        - [Roads](#roads)
    - [Models](#models)
- [Conclusions](#conclusions)
- [Limitations](#limitations)
- [References](#references)
    - [Data](#data)

# Project Proposal

## The Problem

Traffic accidents are common, and can range from minor fender-benders to tragic events that leave people permanently disabled or killed. Even when accidents do not cause physical harm, the financial costs associated with repairing and replacing damaged property can be significant. In addition, there are public costs including emergency personnel and equipment, as well as additional delays and congestion caused by the accident. If the factors that cause traffic accidents could be identified and quantified, extra care could be taken by public institutions to warn drivers and pedestrians of increased dangers, at best preventing accidents before they happen, or at least lessening the impact of accidents on the immediate vicinity through more strategic deployment of first responders.

## Potential Clients

Local governments would be able to make use of a method for prediciting traffic accidents. Emergency services must deal with accidents daily in many jurisdictions, in addition to other situations which threaten life and property. Being able to better position personnel and supplies to deal with traffic accidents, or even implement communications campaigns to promote safety under particularly high-risk circumstances, would benefit the public as well as the specific individuals who are involved in any given accident.

## Data

I plan to use publicly available datasets, primarily from https://data.nola.gov, for this project. In particular, the [Calls for Service](https://data.nola.gov/Public-Safety-and-Preparedness/Calls-for-Service-2012/rv3g-ypg7) data, indicating all emergency response events in New Orleans, datasets will provide the basis for this analysis. Other relevant data from the City of New Orleans includes geospatial and other reference data on the streets of the city. In addition, I will include information on weather conditions such as temperature, precipitation, and wind-speed from [NOAA](https://www1.ncdc.noaa.gov/pub/data/cdo/) and datetime features such as daylight and holidays from https://www.timeanddate.com/.

In addition, an excellent [reference](http://urbcomp.ist.psu.edu/2017/papers/Predicting.pdf) for this analysis comes from a group of researchers who developed a state-wide model for predicting traffic accidents in Iowa.

## Problem Solving Approach

My goal will be to build a binary prediction model, classifying any given road segment and day/hour combination as either having an accident or being accident-free. I will combine the road information with the locations and times of accidents from the *Calls for Service* data, and then append weather and temporal features as appropriate. I plan to use a Random Forest Classifier model and, if time permits, a neural network.

## Deliverables

For this project, I will provide the code used to perform the data cleaning and analysis, as well as a slide deck providing a visual overview of the project and summarizing the findings and any recommendations for potential clients.

# Methods

## Inital Investigation

I intially thought to add additional data related to other activities in the city (all also available on https://data.nola.gov, such as information from police reports, 311 calls, and Department of Public Works Roadwork Projects.

### Codes

```python
target_types = ['100', '100C', '100F', '100I', '100X', '20', '20C', '20F', '20I', '20X']

target_dispositions = ['NAT', 'RTF', 'GOA']
```

## Data Wrangling

```python
intersections = accidents[accidents.BLOCK_ADDRESS.str.contains('&', na=False)].BLOCK_ADDRESS.str.split('&')
intersection_street_1 = intersections.str.get(0).str.strip()
intersection_street_2 = intersections.str.get(1).str.strip()

single_street = accidents[np.logical_not(accidents.BLOCK_ADDRESS.str.contains('&', na=True))].BLOCK_ADDRESS
single_street_1 = single_street[single_street.str.split(' ', n=1).str.get(0).str.contains('X', na=False)].str.split(' ', n=1).str.get(1).str.strip()
single_street_2 = single_street[np.logical_not(single_street.str.split(' ', n=1).str.get(0).str.contains('X', na=True))].str.strip()
```

- Road Length
- Road Type ('Class')
    - Local (y/n)
    - Major Arterial (y/n)
    - Umimproved (y/n)
    - Ramp (y/n)
    - Freeway (y/n)
    - Other (y/n)

1. Spatial Join
2. Keep single matches, further process multiple matches
3. Find centroid of streets, match to neighborhood, then rematch with street geometry
4. Concat all final matches, then dissolve block-by-block street segments into neighborhood-size lengths

- Date
- Hour
- Light (y/n)
- Holiday (y/n)
- Rush Hour (y/n)

```python
relevant_holidays = [
    "New Year's Day",
    "'New Year's Day' observed",
    'Martin Luther King Jr. Day',
    "Valentine's Day",
    "Presidents' Day (Most regions)",
    'Ash Wednesday',
    "St. Patrick's Day",
    'Palm Sunday',
    'Maundy Thursday',
    'Good Friday (Many regions)',
    'Holy Saturday',
    'Easter Sunday',
    'Easter Monday',
    'Cinco de Mayo',
    "Mother's Day",
    'Memorial Day',
    "Father's Day",
    'Independence Day',
    'Labor Day',
    'Columbus Day (Most regions)',
    'Halloween',
    "All Saints' Day",
    "All Souls' Day",
    'Election Day',
    'Veterans Day',
    'Thanksgiving Day',
    'Black Friday',
    'Christmas Eve',
    'Christmas Day',
    "New Year's Eve",
    "'Independence Day' observed",
    'Day After Christmas Day',
    "'Christmas Day' observed",
    'Veterans Day observed']

ash_wednesdays = holidays[holidays['holiday'] == 'Ash Wednesday']
mardi_gras = ash_wednesdays.date + pd.Timedelta('-1 day')
lundi_gras = ash_wednesdays.date + pd.Timedelta('-2 days')
mardi_gras_sunday = ash_wednesdays.date + pd.Timedelta('-3 days')
mardi_gras_saturday = ash_wednesdays.date + pd.Timedelta('-4 days')
mardi_gras_friday = ash_wednesdays.date + pd.Timedelta('-5 days')
mardi_gras_thursday = ash_wednesdays.date + pd.Timedelta('-6 days')
```

- Temperature
- Wind Speed
- Precipitation Total

```python
weather_airport_by_hour = weather_airport.resample('1h').max()

weather_airport_by_hour['precipitation'] = weather_airport_by_hour.precipitation.fillna(0)
weather_airport_by_hour['temp'] = weather_airport_by_hour.temp.interpolate()
weather_airport_by_hour['wind_speed'] = weather_airport_by_hour.wind_speed.interpolate()
```

```python
neighborhoods = gpd.read_file('../../data/raw/Neighborhood_Statistical_Areas/Neighborhood_Statistical_Areas.shp')
neighborhoods = neighborhoods.to_crs(epsg=4326)
neighborhoods = neighborhoods[['OBJECTID', 'GNOCDC_LAB', 'geometry']]
neighborhoods.columns = ['nhood_id', 'nhood', 'geometry']

accidents = pd.read_pickle('../../data/interim/features/accidents.pickle')
accidents['geometry'] = list(zip(accidents.lon, accidents.lat))
accidents['geometry'] = accidents.geometry.apply(shp.geometry.Point)
accidents = gpd.GeoDataFrame(accidents, geometry='geometry')
accidents.crs = {'init' :'epsg:4326'}

accidents_in_neighborhoods = gpd.sjoin(accidents, neighborhoods)

accidents_in_neighborhoods['ItemNumber'] = accidents_in_neighborhoods.index

columns = ['ItemNumber', 'TimeCreate', 'street_1', 'street_2', 'geometry', 'nhood_id', 'nhood']
accidents_in_neighborhoods = accidents_in_neighborhoods[columns]

accidents_join_street = pd.merge(accidents_in_neighborhoods,
                                 streets,
                                 left_on=['street_1', 'nhood_id'],
                                 right_on=['fullnameab', 'nhood_id'])

# select unmatched accidents
unmatched_accidents = accidents_in_neighborhoods[~accidents_in_neighborhoods.index.isin(accidents_join_street.ItemNumber)]

rematched_accidents = pd.merge(unmatched_accidents,
                               streets,
                               left_on=['street_2', 'nhood_id'],
                               right_on=['fullnameab', 'nhood_id'])

no_street_match = unmatched_accidents[~unmatched_accidents.index.isin(rematched_accidents.ItemNumber)]

street_match = pd.concat([accidents_join_street, rematched_accidents], ignore_index=True)
street_match = street_match[['ItemNumber', 'TimeCreate', 'segment_id']]

no_street_match = no_street_match.to_crs(crs_louisiana)

unmatched_items = no_street_match.index

no_street_match['buffer_50'] = no_street_match.geometry.buffer(50)
no_street_match['buffer_100'] = no_street_match.geometry.buffer(100)
no_street_match['buffer_250'] = no_street_match.geometry.buffer(250)

buffer_50 = no_street_match.set_geometry('buffer_50')
buffer_100 = no_street_match.set_geometry('buffer_100')
buffer_250 = no_street_match.set_geometry('buffer_250')

streets_la_crs = streets.to_crs(crs_louisiana)

spatial_join_50 = gpd.sjoin(buffer_50, streets_la_crs)
spatial_join_100 = gpd.sjoin(buffer_100, streets_la_crs)
spatial_join_250 = gpd.sjoin(buffer_250, streets_la_crs)

nhood_match_50 = spatial_join_50[spatial_join_50.nhood_id_left == spatial_join_50.nhood_id_right]

nhood_match_50_unique = nhood_match_50.groupby('ItemNumber').first()
nhood_match_50_unique = nhood_match_50_unique.reset_index()
nhood_match_50_unique = nhood_match_50_unique[['ItemNumber', 'TimeCreate', 'segment_id']]

unmatched_items = unmatched_items[~unmatched_items.isin(nhood_match_50_unique.ItemNumber)]

spatial_join_100_unmatched = spatial_join_100[spatial_join_100.index.isin(unmatched_items)]
nhood_match_100 = spatial_join_100_unmatched[spatial_join_100_unmatched.nhood_id_left == spatial_join_100_unmatched.nhood_id_right]

nhood_match_100_unique = nhood_match_100.groupby('ItemNumber').first()
nhood_match_100_unique = nhood_match_100_unique.reset_index()
nhood_match_100_unique = nhood_match_100_unique[['ItemNumber', 'TimeCreate', 'segment_id']]

unmatched_items = unmatched_items[~unmatched_items.isin(nhood_match_100_unique.ItemNumber)]

spatial_join_250_unmatched = spatial_join_250[spatial_join_250.index.isin(unmatched_items)]
nhood_match_250 = spatial_join_250_unmatched[spatial_join_250_unmatched.nhood_id_left == spatial_join_250_unmatched.nhood_id_right]

nhood_match_250_unique = nhood_match_250.groupby('ItemNumber').first()
nhood_match_250_unique = nhood_match_250_unique.reset_index()
nhood_match_250_unique = nhood_match_250_unique[['ItemNumber', 'TimeCreate', 'segment_id']]

dataframes = [street_match,
              nhood_match_50_unique,
              nhood_match_100_unique,
              nhood_match_250_unique]

joined_matches = pd.concat(dataframes, ignore_index=True)

street_by_time = joined_matches.groupby(['segment_id', 'TimeCreate']).count().reset_index()

street_by_time['day_hour'] = street_by_time.TimeCreate.dt.date.astype('str') \
                             + ' ' \
                             + street_by_time.TimeCreate.dt.hour.astype('str') \
                             + ':00:00'

street_by_time['day_hour'] = pd.to_datetime(street_by_time.day_hour)

street_by_time = street_by_time[['day_hour', 'segment_id', 'ItemNumber']]
street_by_time.columns = ['day_hour', 'segment_id', 'accident_yn']

street_by_time.to_pickle('../../data/interim/features/street_by_time.pickle')
```

**Goal: Dataset for all accidents that is indexed by ROAD SEGMENT and DATETIME**

1. Merge information from other datasets into accidents dataset
2. Select 3 negative samples for each positive (accident) sample according to methodology from article
3. Append negative samples to positive samples and clean up final output dataset

```python
np.random.seed(42)
sample_space_date = pd.Series(pd.date_range('2012-01-01', '2018-06-30')).dt.date
sample_space_hour = pd.Series([x for x in range(24)])
sample_space_location = street_info.segment_id.sort_values().reset_index(drop=True)
```


```python
n_rows = 1000000
sample_dates = np.random.choice(sample_space_date, size=n_rows)
sample_hours = np.random.choice(sample_space_hour, size=n_rows)
sample_locations = np.random.choice(sample_space_location, size=n_rows)
```

```python
renamed_cols = ['date',
                'hour',
                'segment_id',
                'daylight_yn',
                'holiday_yn',
                'rush_hour_yn',
                'temp',
                'wind_speed',
                'precipitation',
                'road_length',
                'class_freeway',
                'class_local',
                'class_major',
                'class_other',
                'class_unimproved',
                'accident_yn']
```

# Results

## Statistics/Visuals

```python
streets = gpd.read_file('../../data/raw/Road_Centerline/geo_export_c02761a8-1d85-477e-a5cf-01b9f22f4d88.shp')
street_columns = ['joinid', 'roadclass', 'fullname', 'geometry']
streets = streets[street_columns]
streets = streets.dissolve(by='joinid', aggfunc='first')
streets['roadclass'] = streets.roadclass.fillna('Other')

neighborhoods = gpd.read_file('../../data/raw/Neighborhood_Statistical_Areas/Neighborhood_Statistical_Areas.shp')
nhood_columns = ['OBJECTID', 'GNOCDC_LAB', 'geometry']
neighborhoods = neighborhoods[nhood_columns]
neighborhoods.columns = ['nhood_id', 'nhood', 'geometry']
neighborhoods.set_index('nhood_id', inplace=True)
neighborhoods = neighborhoods.to_crs(epsg=4326)

sample_data = pd.read_pickle('../../data/processed/all_samples.pickle')
sample_data['datetime'] = pd.to_datetime(sample_data.date)

accidents = pd.read_pickle('../../data/interim/features/accidents.pickle')
accidents['geometry'] = list(zip(accidents.lon, accidents.lat))
accidents['geometry'] = accidents.geometry.apply(shp.geometry.Point)
accidents = gpd.GeoDataFrame(accidents, geometry='geometry')
accidents.crs = {'init' :'epsg:4326'}
```

### Maps


```python
figure_size = (20, 20)

nhood_base = neighborhoods.plot(figsize=figure_size,
                                color='gray',
                                edgecolor='black',
                                alpha=.5)

streets_local = streets[streets['roadclass'] == 'Local'].plot(ax=nhood_base,
                                                              figsize=figure_size,
                                                              color='white',
                                                              alpha=.2)

streets_major = streets[streets['roadclass'] == 'Major Arterial'].plot(ax=streets_local,
                                                                       figsize=figure_size,
                                                                       color='green',
                                                                       alpha=.4)

streets_freeway = streets[streets['roadclass'] == 'Freeway'].plot(ax=streets_major,
                                                                  figsize=figure_size,
                                                                  color='blue',
                                                                  alpha=.5)
```


![png](output_3_0.png)



```python
nhood_base = neighborhoods.plot(figsize=figure_size,
                                color='gray',
                                edgecolor='gray',
                                alpha=.5)

accidents.plot(ax=nhood_base, marker='*', color='red', markersize=1, alpha=.2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x16ba175f710>




![png](output_4_1.png)


### Charts


```python
plt.style.use('ggplot')
plt.rcParams['figure.figsize'][0] = 16
plt.rcParams['figure.figsize'][1] = 9
```


```python
accident_samples = sample_data[sample_data['accident_yn'] == 1].copy()
```


```python
months = ['Jan', 'Feb', 'Mar', 'Apr','May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
accident_samples['month'] = pd.Categorical(accident_samples.datetime.dt.strftime('%b'), categories=months)

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
accident_samples['day'] = pd.Categorical(accident_samples.datetime.dt.strftime('%a'), categories=days)
```

### Accident Counts by Date

**Month**


```python
_ = accident_samples.groupby(['month']).size().plot(kind='bar')
```


![png](output_11_0.png)


**Day of Week**


```python
_ = accident_samples.groupby(['day']).size().plot(kind='bar')
```


![png](output_13_0.png)


**Hour of Day**


```python
accidents_by_day_hour = accident_samples.groupby(['day', 'hour']).size().reset_index()
accidents_by_day_hour.columns = ['day', 'hour', 'accidents']

accidents_by_day_hour_pivoted = accidents_by_day_hour.pivot('hour', 'day', 'accidents')

f, ax = plt.subplots(figsize=(20, 16))

_ = sns.heatmap(accidents_by_day_hour_pivoted, annot=True, fmt='d', linewidths=.5, ax=ax)
```


![png](output_15_0.png)


**Daylight**


```python
_ = accident_samples.groupby(['daylight_yn']).size().plot(kind='bar')
```


![png](output_17_0.png)


**Holiday**


```python
_ = accident_samples.groupby(['holiday_yn']).size().plot(kind='bar')
```


![png](output_19_0.png)


**Rush Hour**


```python
_ = accident_samples.groupby(['rush_hour_yn']).size().plot(kind='bar')
```


![png](output_21_0.png)


### Weather Distribution

**Temperature**


```python
_ = plt.hist(accident_samples.temp, bins=16)
```


![png](output_24_0.png)


**Wind**


```python
_ = plt.hist(accident_samples.wind_speed, bins=25)
```


![png](output_26_0.png)


**Precipitation**


```python
rain_plot = accident_samples[(accident_samples.precipitation > 0) & (accident_samples.precipitation < .5)].precipitation
_ = plt.hist(rain_plot, bins=25)
```


![png](output_28_0.png)


### Roads

**Type**


```python
road_classes = ['class_freeway', 'class_local', 'class_major', 'class_other', 'class_unimproved']

_ = accident_samples[road_classes].idxmax(axis=1).value_counts().plot(kind='bar')
```


![png](output_31_0.png)


**Length**


```python
_ = plt.subplot(2, 1, 1)
_ = plt.hist(accident_samples[accident_samples.road_length < 5280].road_length, bins=20)
_ = plt.title('Road Length (less than 1-mile)')
_ = plt.subplot(2, 1, 2)
_ = plt.hist(accident_samples[accident_samples.road_length >= 5280].road_length, bins=30)
_ = plt.title('Road Length (greater than 1-mile)')

plt.rcParams['figure.figsize'][0] = 16
plt.rcParams['figure.figsize'][1] = 20
```


![png](output_33_0.png)



## Models

```python
features = ['daylight_yn',
            'holiday_yn',
            'rush_hour_yn',
            'temp',
            'wind_speed',
            'precipitation',
            'road_length',
            'class_freeway',
            'class_local',
            'class_major',
            'class_other',
            'class_unimproved']

labels = 'accident_yn'
```


```python
X = data[features]
y = data[labels]
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
```


```python
forest_100 = RandomForestClassifier(n_estimators=100)

forest_100.fit(X_train, y_train)

y_pred_100 = forest_100.predict(X_test)

print('Random Forest (n=100)')

print('Accuracy:', metrics.accuracy_score(y_test, y_pred_100))
print('Precision:', metrics.precision_score(y_test, y_pred_100))
print('Recall:', metrics.recall_score(y_test, y_pred_100))

feature_importance_100 = pd.Series(forest_100.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importance_100)
```

    Random Forest (n=100)
    Accuracy: 0.8211645646353579
    Precision: 0.656999452799215
    Recall: 0.5889247839250377
    road_length         0.550893
    temp                0.163282
    class_major         0.087130
    wind_speed          0.078128
    class_local         0.061115
    daylight_yn         0.018803
    precipitation       0.013709
    class_freeway       0.009210
    rush_hour_yn        0.007006
    class_unimproved    0.006917
    holiday_yn          0.003110
    class_other         0.000698
    dtype: float64


# Conclusions



# Limitations



# References

## Data

- [Weather](https://www1.ncdc.noaa.gov/pub/data/cdo/documentation/LCD_documentation.pdf)
- [Calls for Service 2012](https://data.nola.gov/Public-Safety-and-Preparedness/Calls-for-Service-2012/rv3g-ypg7)
- [Calls for Service 2013](https://data.nola.gov/Public-Safety-and-Preparedness/Calls-for-Service-2013/5fn8-vtui)
- [Calls for Service 2014](https://data.nola.gov/Public-Safety-and-Preparedness/Calls-for-Service-2014/jsyu-nz5r)
- [Calls for Service 2015](https://data.nola.gov/Public-Safety-and-Preparedness/Calls-for-Service-2015/w68y-xmk6)
- [Calls for Service 2016](https://data.nola.gov/Public-Safety-and-Preparedness/Calls-for-Service-2016/wgrp-d3ma)
- [Calls for Service 2017](https://data.nola.gov/Public-Safety-and-Preparedness/Calls-for-Service-2017/bqmt-f3jk)
- [Calls for Service 2018](https://data.nola.gov/Public-Safety-and-Preparedness/Calls-for-Service-2018/9san-ivhk)
- [Neighborhoods](https://data.nola.gov/Geographic-Base-Layers/Neighborhood-Statistical-Areas/c2j2-5qdf)
- [Holiday and Date/Time](https://www.timeanddate.com/)