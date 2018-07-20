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
    - [Data Wrangling](#data-wrangling)
- [Results](#results)
    - [Inferential Statistics](#inferential-statistics)
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

## Data Wrangling



# Results

## Inferential Statistics



## Models



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