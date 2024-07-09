from configuration import BASE_GOWALLA, CHECKINS, SPOTS_1, CATEGORY_STRUCTURE, TIMEZONES, BRASIL_STATES, COUNTRIES, \
    CHECKINS_LOCAL_DATETIME, CHECKINS_7_CATEGORIES

import json
import numpy as np
import pandas as pd
import ast


def to_super_category(categories, to_super_category_dict):
    super_categories = []
    outras = []
    n = 0
    for e in categories:
        e = e.replace("[", "").replace("]", "")
        e = ast.literal_eval(e)
        category = e['name']
        try:
            category = to_super_category_dict[category]
            super_categories.append(category)
        except:
            super_categories.append(np.nan)
            outras.append(category)
            n += 1

    return np.array(super_categories)


if __name__ == "__main__":

    checkins = pd.read_csv(CHECKINS)
    checkins = checkins.sample(frac=0.1)

    pois1 = pd.read_csv(SPOTS_1)[['id', 'lat', 'lng', 'spot_categories']]
    pois1.columns = ['placeid', 'latitude', 'longitude', 'category']

    checkins = checkins.join(pois1.set_index('placeid'), on='placeid').dropna()

    with open(CATEGORY_STRUCTURE) as f:
        data = json.load(f)

    super_categories_dict = {'Other - Travel & Lodging': 'Travel', 'Urban Outfitters': 'Shopping', 'Starbucks': 'Food',
                             'Walmart': 'Shopping', 'Nike': 'Shopping',
                             'Gap': 'Shopping', 'Apple Store': 'Shopping', 'Best Buy': 'Shopping',
                             'Whole Foods': 'Shopping', 'Target': 'Shopping', 'Nightlife': 'Nightlife',
                             'T-Mobile': 'Shopping', 'In-N-Out Burger': 'Food', 'Five Guys Burgers and Fries': 'Food',
                             'Four Seasons': 'Travel', 'Holiday Inn': 'Travel',
                             "McDonald's": 'Food', 'Holiday Inn Express': 'Travel', 'Walgreens': 'Shopping',
                             'Walt Disne World Resort': 'Travel', 'Burger King': 'Food',
                             'Holiday Inn Club Vacations': 'Travel', 'Chipotle': 'Food', 'Staybridge Suites': 'Travel',
                             'Banana Republic': 'Shopping', 'AMC Theatre': 'Entertainment',
                             "Dukin' Donuts": 'Food', 'Other - Art & Culture': 'Entertainment',
                             "Food & Foodies": "Food", "Cinemark Theatre": 'Entertainment',
                             'Other - Parks ': 'Outdoors'}
    for e in data['spot_categories']:
        super_category = e['name']
        super_url = e['url']
        inner_categories = e['spot_categories']
        for inner in inner_categories:
            inner_url = inner['url']
            inner_category = inner['name']
            inner_inner_categories = inner['spot_categories']

            for inner_inner in inner_inner_categories:
                inner_inner_url = inner_inner['url']
                inner_inner_category = inner_inner['name']

                super_categories_dict[inner_inner_category] = super_category

            super_categories_dict[inner_category] = super_category

    checkins['category'] = to_super_category(checkins['category'].tolist(), super_categories_dict)

    print("categorias únicas: ")
    print(checkins['category'].unique().tolist())
    checkins = checkins.dropna()

    checkins.to_csv(CHECKINS_7_CATEGORIES, index_label=False, index=False)
    print("Finished")