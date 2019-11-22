import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from PIL import Image

# for testing purposes, remove this later!
from sys import exit

"""Data visualization on the Airbnb New York dataset from Kaggle.

The dataset provides 16 pieces of data in the following order:

0: id
1: name
2: host_id
3: host_name
4: neighbourhood_group
5: neighbourhood
6: latitude
7: longitude
8: room_type
9: price
10: minimum_nights
11: number_of_reviews
12: last_review
13: reviews_per_month
14: calculated_host_listings_count
15: availability_365

All fields are fairly self-explanatory. I will not be using the 'id' or the
'host_id' field since they are not relevant, and the 'name' field since it does
not make sense to in this context.

This project is fully open source and free to use and share. Enjoy!
"""

header = []
data = {}
num_columns = 16
num_entries = 0

with open('new_york_data.csv', encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    # read the header
    header = next(reader)
    # read the entries
    body = []
    for row in reader:
        body.append(row)
    num_entries = len(body)
    # parse the entries into np arrays and store them under in the data list
    for i in range(num_columns):
        dtype = 'str'
        # price, minimum nights, number of reviews
        # calculated host listings count, annual availability
        if i == 9 or i == 10 or i == 11 or i == 14 or i == 15:
            dtype = 'int64'
        # latitude, longitude, review per month
        if i == 6 or i == 7 or i == 13:
            dtype = 'float64'
        # reviews per month is blank sometimes in the original dataset
        if i == 13:
            # numpy cannot process empty strings to floats; so check for this
            col_data = np.asarray([body[j][i] if len(body[j][i]) > 0 else 0.0 for j in range(num_entries)], dtype=dtype)
        else:
            col_data = np.asarray([body[j][i] for j in range(num_entries)], dtype=dtype)
        data[header[i]] = col_data

# Area that the cover maps; experimentally determined
# (latitude, longitude)
min_coords = (40.49279, -74.26442)
max_coords = (40.91906, -73.68299)
long_range = max_coords[1] - min_coords[1]
lat_range = max_coords[0] - min_coords[0]
image_extent = (min_coords[1], max_coords[1], min_coords[0], max_coords[0])
new_york_img = Image.open('new_york_map.png')

# use large figure sizes
matplotlib.rcParams['figure.figsize'] = (12, 7)

# Room Type Bar Graph
room_types, room_types_count = np.unique(data['room_type'], return_counts=True)
plt.title('Distribution of Room Types')
room_types_norm = room_types_count / sum(room_types_count)
plt.barh(room_types, room_types_norm)
ax = plt.gca()
ax.xaxis.set_major_formatter(tck.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
plt.show()

# Neighbourhood Groups
n_groups, n_groups_count = np.unique(data['neighbourhood_group'], return_counts=True)
n_groups_colors = ['#1a535c', '#4ecdc4', '#b2ff66', '#ff6b6b', '#ffe66d']

explode = np.zeros((len(n_groups),), dtype='float64')
for idx, group in enumerate(n_groups):
    if group == 'Manhattan':
        explode[idx] = 0.1
        break
plt.title('Distribution of Neighbourhood Groups')
wedges, texts, _ = plt.pie(
    n_groups_count,
    labels=n_groups,
    explode=explode,
    autopct='%1.1f%%',
    pctdistance=0.8,
    colors=n_groups_colors)
plt.show()

# Neighbourhoods
nbhs, nbhs_count = np.unique(data['neighbourhood'], return_counts=True)
# zip the neighbourhood name and count into a tuple to sort by count
nbhs_sorted_tuples = sorted(list(zip(nbhs, nbhs_count)), key=lambda elem: elem[1], reverse=True)
# unzip the sorted tuples back into a list of names and a list of counts
nbhs_sorted, nbhs_sorted_count = list(zip(*nbhs_sorted_tuples))
# take only the top 20
nbhs_sorted = nbhs_sorted[:20]
nbhs_sorted_count = nbhs_sorted_count[:20]
nbhs_price_avgs = []
for nbh in nbhs_sorted:
    prices = data['price'][data['neighbourhood'] == nbh]
    nbhs_price_avgs.append(np.average(prices))
fig, ax1 = plt.subplots()
plt.title('Most Popular Neighbourhoods and Average Price')
# pad the bottom of the plot to prevent text clipping
plt.subplots_adjust(bottom=0.2)
# rotate the labels so that they are easier to read
ax1.set_xticklabels(nbhs_sorted, rotation=45, ha='right')
ax1.set_xlabel('Neighbourhood');
# plot number of places on the left y-axis
ax1.bar(nbhs_sorted, nbhs_sorted_count, width=-0.2, align='edge')
ax1.set_ylabel('Number of places (blue)')
# plot average price on the right y-axis
ax2 = ax1.twinx()
ax2.bar(nbhs_sorted, nbhs_price_avgs, width=0.2, align='edge', color='orange')
ax2.set_ylabel('Average price (orange)')
plt.show()

# Price Histogram
group_prices = []
# separate the price data based on neighbourhood groups
for group in n_groups:
    group_prices.append(data['price'][data['neighbourhood_group'] == group])
# plot the price data for each group separately as stacked bars
# use only prices less than 500 since most of the data belongs in this range
# this also lets us not worry about huge outliers (there are a few places whose
# nightly price is in the many thousands)
plt.hist(
    group_prices,
    histtype='barstacked',
    bins=25,
    range=(0, 500),
    edgecolor='white',
    color=n_groups_colors)
plt.legend(n_groups, loc='upper right')
plt.title('Distribution of Price per Night')
plt.xlim(0, 500)
plt.ylabel('Number of places')
plt.xlabel('Price range (USD)')
plt.show()

# Average Price Heatmap
# compute the average pricing over a grid of 150 by 150
price_heatmap_bins = 150
price_heatmap_sum = np.zeros((price_heatmap_bins, price_heatmap_bins), dtype='float64')
price_heatmap_count = np.zeros((price_heatmap_bins, price_heatmap_bins), dtype='float64')
for long, lat, price in zip(data['longitude'], data['latitude'], data['price']):
    # take only prices below 500 to be consistent with price histogram
    if price < 500:
        idx_long = int((long - min_coords[1]) / long_range * price_heatmap_bins)
        idx_lat = int((lat - min_coords[0]) / lat_range * price_heatmap_bins)
        price_heatmap_sum[idx_lat, idx_long] += price
        price_heatmap_count[idx_lat, idx_long] += 1
# ensure that a divide by zero will not occur
price_heatmap_count = np.clip(price_heatmap_count, 1, None)
price_heatmap = price_heatmap_sum / price_heatmap_count
plt.imshow(new_york_img, extent=image_extent)
plt.imshow(price_heatmap, extent=image_extent, origin='lower', alpha=0.9)
plt.colorbar()
plt.title('Average Price per Night Heatmap')
plt.show()

# Housing Scatter Plot
plt.imshow(new_york_img, extent=image_extent)
# divide locations based on groups and display them as a scatter on the New York map
for group, color in zip(n_groups, n_groups_colors):
    plt.scatter(
        data['longitude'][data['neighbourhood_group'] == group],
        data['latitude'][data['neighbourhood_group'] == group],
        s=2,
        color=color)
plt.legend(n_groups, loc='upper left', markerscale=5)
plt.title('Plot of Housing Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Housing Heatmap
plt.imshow(new_york_img, extent=image_extent)
plt.hist2d(data['longitude'], data['latitude'], bins=150, alpha=0.7)
plt.title('Heatmap of Housing Locations')
plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Minimum Nights Distribution
group_min_nights = []
# separate the price data based on neighbourhood groups
for group in n_groups:
    group_min_nights.append(data['minimum_nights'][data['neighbourhood_group'] == group])
# plot the price data for each group separately as stacked bars
plt.hist(
    group_min_nights,
    histtype='barstacked',
    bins=20,
    range=(1, 21),
    edgecolor='white',
    color=n_groups_colors)
plt.title('Minimum Number of Nights Required')
plt.legend(n_groups, loc='upper right')
plt.xlim(1, 21)
plt.xticks(np.arange(1, 21))
plt.xlabel('Minimum Nights')
plt.ylabel('Number of Places')
plt.show()

# Number of Reviews
# compute the average number of reviews over a grid of 150 by 150
num_reviews_bins = 150
num_reviews_sum = np.zeros((num_reviews_bins, num_reviews_bins), dtype='float64')
num_reviews_count = np.zeros((num_reviews_bins, num_reviews_bins), dtype='float64')
for long, lat, price in zip(data['longitude'], data['latitude'], data['number_of_reviews']):
    idx_long = int((long - min_coords[1]) / long_range * num_reviews_bins)
    idx_lat = int((lat - min_coords[0]) / lat_range * num_reviews_bins)
    num_reviews_sum[idx_lat, idx_long] += price
    num_reviews_count[idx_lat, idx_long] += 1
# ensure that a divide by zero will not occur
num_reviews_count = np.clip(num_reviews_count, 1, None)
num_reviews = num_reviews_sum / num_reviews_count
plt.imshow(new_york_img, extent=image_extent)
plt.imshow(num_reviews, extent=image_extent, origin='lower', alpha=0.9)
plt.colorbar()
plt.title('Average Number of Reviews Heatmap')
plt.show()
