import csv
import numpy as np
import matplotlib.pyplot as plt

"""Data analysis on the Airbnb New York dataset from Kaggle.

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

print('Average price:', sum(data['price'] / num_entries))
